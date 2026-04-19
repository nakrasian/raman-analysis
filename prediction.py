import joblib
import numpy as np
import shap
from scipy.special import softmax

class SpectraClassifier:
    """
    A class used to utilize a pretrained model for classifying spectra, 
    with capabilities for group aggregation and interpretability via SHAP.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        model_pkg = joblib.load(model_path)
        self.scaler = model_pkg['scaler']
        self.model = model_pkg['model']
        
        self.has_predict_proba = hasattr(self.model, "predict_proba")

    def predict_aggregate(self, spectra):
        """
        Scales inputs, performs prediction, and aggregates across all spectra 
        to return a majority vote and confidence percentage.
        """
        spectra = np.asarray(spectra)
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        scaled_spectra = self.scaler.transform(spectra)
        predictions = self.model.predict(scaled_spectra)
        
        # Majority Vote
        values, counts = np.unique(predictions, return_counts=True)
        majority_prediction = values[np.argmax(counts)]
        
        confidence = 0.0
        if len(predictions) > 1:
            # Aggregate Prediction: Confidence is the Majority Score Ratio (Percentage of frames voting for this class)
            confidence = float(np.max(counts)) / float(len(predictions))
        else:
            # Single Spectra Prediction: Confidence is the exact mathematical model probability
            if self.has_predict_proba:
                probs = self.model.predict_proba(scaled_spectra)[0]
                try:
                    class_idx = np.where(self.model.classes_ == majority_prediction)[0][0]
                    confidence = probs[class_idx]
                except Exception:
                    confidence = np.max(probs)
            else:
                if hasattr(self.model, "decision_function"):
                    distances = self.model.decision_function(scaled_spectra)[0]
                    if np.ndim(distances) == 0:
                        prob = 1 / (1 + np.exp(-distances))
                        if majority_prediction == self.model.classes_[1]:
                            confidence = prob
                        else:
                            confidence = 1.0 - prob
                    else:
                        prob = softmax(distances.reshape(1, -1), axis=1)[0]
                        try:
                            class_idx = np.where(self.model.classes_ == majority_prediction)[0][0]
                            confidence = prob[class_idx]
                        except Exception:
                            confidence = np.max(prob)
                else:
                    confidence = 1.0
                    
        return majority_prediction, float(confidence)

    def get_shap_explanations(self, spectra, background_size=15):
        """
        Generates local feature importance values using SHAP.
        Returns a 1D array representing aggregated absolute feature impacts.
        """
        spectra = np.asarray(spectra)
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        scaled_spectra = self.scaler.transform(spectra)
        
        def model_predict_func(x):
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(x)
            elif hasattr(self.model, "decision_function"):
                dist = self.model.decision_function(x)
                if dist.ndim == 1:
                    return 1 / (1 + np.exp(-dist))
                else:
                    return softmax(dist, axis=1)
            else:
                return np.zeros(x.shape[0])
                
        try:
            model_type = type(self.model).__name__
            if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 'DecisionTreeClassifier']:
                try:
                    explainer = shap.TreeExplainer(self.model)
                    shap_values_output = explainer.shap_values(scaled_spectra)
                    
                    if isinstance(shap_values_output, list):
                        v = shap_values_output[-1]
                    else:
                        v = shap_values_output
                        
                    if len(v.shape) == 3:
                        v = v[:, :, -1]
                        
                    mean_impact = np.mean(np.abs(v), axis=0) # Average over samples
                    if len(mean_impact.shape) > 1:
                        mean_impact = np.max(mean_impact, axis=-1)
                    return mean_impact
                except Exception as tree_e:
                    print(f"TreeExplainer failed ({tree_e}), falling back to KernelExplainer.")
                    background = shap.utils.sample(scaled_spectra, min(background_size, scaled_spectra.shape[0]))
                    explainer = shap.KernelExplainer(model_predict_func, background)
                    
                    # Subsample the evaluation group to prevent infinite loading
                    eval_spectra = shap.utils.sample(scaled_spectra, min(5, scaled_spectra.shape[0]))
                    shap_values_output = explainer.shap_values(eval_spectra, silent=True)
                    
                    if isinstance(shap_values_output, list):
                        v = shap_values_output[-1] 
                    else:
                        v = shap_values_output
                        
                    mean_impact = np.mean(np.abs(v), axis=0)
                    if len(mean_impact.shape) > 1:
                        mean_impact = np.max(mean_impact, axis=-1)
                    return mean_impact
                
            else:
                background = shap.utils.sample(scaled_spectra, min(background_size, scaled_spectra.shape[0]))
                explainer = shap.KernelExplainer(model_predict_func, background)
                
                # Subsample the evaluation group to prevent infinite loading
                eval_spectra = shap.utils.sample(scaled_spectra, min(5, scaled_spectra.shape[0]))
                shap_values_output = explainer.shap_values(eval_spectra, silent=True)
                
                if isinstance(shap_values_output, list):
                    v = shap_values_output[-1] 
                else:
                    v = shap_values_output
                    
                mean_impact = np.mean(np.abs(v), axis=0)
                if len(mean_impact.shape) > 1:
                    mean_impact = np.max(mean_impact, axis=-1)
                return mean_impact
                
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return np.zeros(scaled_spectra.shape[1])

import os
import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.express as px

from scipy import sparse
from scipy.sparse.linalg import spsolve

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import joblib
import seaborn as sns
from preprocessing import SpectraPreprocessor


METADATA_FOLDER = "metadata/4-10-2026-metadata"
SPECTRA_FOLDER = "spectra/4-10-2026-spectra"
BASELINE_FOLDER = "plots/baseline"
ANIMATE_FOLDER = "animations/preprocessed"
SNV_FOLDER = "plots/snv"
PRE_PROCESSED_FOLDER = "plots/preprocessed"
SPECTRA_LENGTH = 1024
CROP_THRESHOL = 0
COMPONENT_ANALYSIS_FOLDER = "plots/component_analysis"
CONFUSION_MATRIX_FOLDER = "model/confusion_matrix"
IMPORTANCE_FOLDER = "model/feature_importance"


## Preprocessing Functions
# (Moved to preprocessing.py)


## Diagnostic and Animating Functions
def plot_baseline_diagnostic(wavelengths, intensities, baseline, corrected, label):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Original + Estimated Baseline
    axes[0].plot(wavelengths, intensities, label="Original Signal", color='black', alpha=0.5)
    axes[0].plot(wavelengths, baseline, label="Estimated Baseline", color='red', linestyle='--')
    axes[0].set_title(f"Original & Baseline: {label}")
    axes[0].legend()

    # 2. Corrected Signal (The Result)
    axes[1].plot(wavelengths, corrected, label="Corrected Signal", color='blue')
    axes[1].set_title("Baseline Corrected (Signal - Baseline)")
    axes[1].legend()

    # 3. The Residual (The Baseline itself)
    axes[2].plot(wavelengths, baseline, label="Residual (Background)", color='orange')
    axes[2].set_title("Residual (The part that was removed)")
    axes[2].set_xlabel("Wavenumber (cm⁻¹)")
    axes[2].legend()

    plt.tight_layout()
    diagnostic_file = os.path.join(BASELINE_FOLDER, f"{label}_baseline_diagnostic.png")
    os.mkdir(BASELINE_FOLDER) if not os.path.exists(BASELINE_FOLDER) else None
    plt.savefig(diagnostic_file)
    print(f"Saved baseline diagnostic plot to {diagnostic_file}")
    plt.close()

def plot_normalization_diagnostic(wavenumbers, baselined, normalized, label):
    """
    Plots a comparison between the baselined spectrum and the normalized version.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 1. Post-Baseline (Before Normalization)
    axes[0].plot(wavenumbers, baselined, color='blue', alpha=0.7)
    axes[0].set_title(f"After Baseline Correction: {label}")
    axes[0].set_ylabel("Intensity (Counts)")
    axes[0].grid(True, which='both', linestyle='--', alpha=0.5)

    # 2. After Normalization (SNV)
    axes[1].plot(wavenumbers, normalized, color='green')
    axes[1].set_title("After SNV Normalization")
    axes[1].set_ylabel("Standardized Intensity (Z-score)")
    axes[1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[1].grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    snv_file = os.path.join(SNV_FOLDER, f"{label}_snv_diagnostic.png")
    os.mkdir(SNV_FOLDER) if not os.path.exists(SNV_FOLDER) else None
    plt.savefig(snv_file)
    print(f"Saved SNV diagnostic plot to {snv_file}")
    plt.close()

def animate_preprocessed_spectra(spectra_name, wavelengths, snv_intensities, label=None):
    """
    Animates or plots overlaid SNV-normalized spectra.
    """
    print(f"Generating SNV visualization for {spectra_name}")

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_title(f"SNV Normalized Spectra: {label}")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Standardized Intensity (Z-score)")

    # Adjust limits dynamically based on SNV data
    ax.set_xlim(wavelengths.min(), wavelengths.max())
    
    # SNV data usually sits between -3 and 15. 
    # We add a 10% buffer to the min/max for visibility.
    y_min = np.min(snv_intensities) * 1.1
    y_max = np.max(snv_intensities) * 1.1
    ax.set_ylim(0, 7)
    #ax.set_ylim(y_min, y_max)


    cmap = plt.get_cmap('viridis')
    
    # Static Overlay Plot (All spectra at once)
    for i in range(len(snv_intensities)):
        color = cmap(i / len(snv_intensities))
        ax.plot(wavelengths, snv_intensities[i], color=color, alpha=0.3)

    # Save the static plot
    plot_file = os.path.join(PRE_PROCESSED_FOLDER, f"{spectra_name}_preprocessed_static.png")
    os.makedirs(PRE_PROCESSED_FOLDER, exist_ok=True)
    plt.savefig(plot_file)
    print(f"Saved static preprocessed plot to {plot_file}")

    ax.clear()

    # --- Animation Logic ---
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        # Update line with the SNV intensity of the current frame
        line.set_data(wavelengths, snv_intensities[frame])
        line.set_color(cmap(frame / len(snv_intensities)))
        return line,

    num_frames = len(snv_intensities)
    # Forward and backward for a smooth loop
    frames_bidirectional = np.concatenate([np.arange(num_frames), np.arange(num_frames)[::-1]])

    ani = animation.FuncAnimation(
        fig=fig, 
        func=update, 
        frames=frames_bidirectional, 
        init_func=init, 
        blit=True, 
        interval=50
    )

    animate_file = os.path.join(ANIMATE_FOLDER, f"{spectra_name}_snv_animation.gif")
    os.makedirs(ANIMATE_FOLDER, exist_ok=True)
    
    # Save as GIF
    ani.save(animate_file, writer='pillow')
    print(f"Saved SNV animation to {animate_file}")
    plt.close()

def save_metrics_plot(all_true, all_preds, model_name, groupings, output_folder="model/metrics"):
    os.makedirs(os.path.join(output_folder, groupings), exist_ok=True)

    # Calculate metrics using 'weighted' so it works safely for both binary and multi-class
    acc = accuracy_score(all_true, all_preds)
    prec = precision_score(all_true, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_true, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_true, all_preds, average='weighted', zero_division=0)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [acc, prec, rec, f1]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(metrics, scores, color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'])
    plt.ylim(0, 1.1) # Set y-axis from 0 to 1.1 to make room for labels
    plt.title(f"Overall Performance: {model_name}")
    plt.ylabel("Score")

    # Add the exact numbers on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, 
                 f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_folder, groupings, f"{model_name}_metrics_bar.png")
    plt.savefig(save_path, dpi=300) # dpi=300 makes it high-resolution
    print(f"Saved metrics bar chart to {save_path}")
    plt.close()
## Component Analysis Functions
def run_pca_analysis(X, y, sample_names, groupings):
    print("Running PCA analysis...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    var_explained = pca.explained_variance_ratio_ * 100

    pc1_label = f"PC1 ({var_explained[0]:.1f}% var)"
    pc2_label = f"PC2 ({var_explained[1]:.1f}% var)"

    df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Target': y,
        'Sample_ID': sample_names
    })
    
    # Plotly does the heavy lifting
    fig = px.scatter(
        df, x='PC1', y='PC2', 
        color='Target', 
        hover_name='Sample_ID', # This is the magic interactive part
        title='Interactive PCA of SERS Spectra',
        opacity=0.7,
        labels={'PC1': pc1_label, 'PC2': pc2_label, 'Target': 'Spectra Group'}
    )
    html_file = os.path.join(COMPONENT_ANALYSIS_FOLDER, groupings, "pca.html")
    os.makedirs(os.path.join(COMPONENT_ANALYSIS_FOLDER, groupings), exist_ok=True)
    fig.write_html(html_file)
    print(f"Saved PCA plot to {html_file}")
    # fig.show()

def run_tsne_analysis(X, y, sample_names, groupings, perplexity=15):
    print(f"Running Interactive t-SNE (Perplexity: {perplexity})...")
    
    # Initialize and fit t-SNE. random_state ensures reproducible plots.
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Create DataFrame
    df = pd.DataFrame({
        't-SNE 1': X_tsne[:, 0],
        't-SNE 2': X_tsne[:, 1],
        'Target': y,
        'Sample_ID': sample_names
    })
    
    # Plotly Scatter
    fig = px.scatter(
        df, 
        x='t-SNE 1', 
        y='t-SNE 2', 
        color='Target', 
        hover_name='Sample_ID', 
        title=f'Interactive t-SNE of SERS Spectra (Perplexity: {perplexity})',
        opacity=0.8,
        labels={'Target': 'Spectra Group'}
    )
    
    # Update layout to look clean
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False)
    )
    html_file = os.path.join(COMPONENT_ANALYSIS_FOLDER, groupings, f"tsne_perplexity_{perplexity}_{groupings}.html")
    os.makedirs(os.path.join(COMPONENT_ANALYSIS_FOLDER, groupings), exist_ok=True)
    fig.write_html(html_file)
    print(f"Saved t-SNE plot to {html_file}")
    # fig.show()

## Main Call for Preprocessing
def preprocess_data(wavelengths, intensities, label):
    # Placeholder for data preprocessing code
    print("Preprocessing data...")
    baseline_spectra = []

    sample_y = intensities[0]
    baseline_y = SpectraPreprocessor.baseline_als(sample_y)
    corrected_y = sample_y - baseline_y

    # Preprocess each spectrum: Baseline correction followed by SNV normalization
    for spectrum in intensities:
        baseline = SpectraPreprocessor.baseline_als(spectrum)
        corrected = spectrum - baseline
        baseline_spectra.append(corrected)
    
    pre_processed_spectra = SpectraPreprocessor.apply_snv(baseline_spectra)

    ## Diagnostics and Animations Plot
    # plot_baseline_diagnostic(wavelengths, sample_y, baseline_y, corrected_y, label)
    # plot_normalization_diagnostic(wavelengths, baseline_spectra[0], pre_processed_spectra[0], label)
    # animate_preprocessed_spectra(label, wavelengths, pre_processed_spectra, label)
    return np.array(pre_processed_spectra)


## Reads Spectra and Metadata Files
def read_spectra(spectra_path, metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = f.read()


    # Extracting metadata information
    num_spectra = int(metadata.splitlines()[0].split(":")[1].strip())
    label = metadata.splitlines()[6].split(":")[1].strip()
    group = metadata.splitlines()[7].split(":")[1].strip()

    spectra = pd.read_csv(spectra_path, header=None, names=["Wavenumber", "Intensity"])
    wavelengths = spectra["Wavenumber"].head(SPECTRA_LENGTH).values
    intensities = spectra["Intensity"].values.reshape(num_spectra, SPECTRA_LENGTH)

    print (f"Number of spectra: {num_spectra}")
    print (f"Label: {label}")
    print (f"Group: {group}")
    print (f"Wavelengths: {wavelengths[:5]} ...")
    print (f"Intensities shape: {intensities.shape}")

    return wavelengths, intensities, label, group

def save_final_model(X, y, wavelengths, model, model_name, output_folder="model/saved_models", groupings="Coffee"):
    # 1. Initialize and fit the scaler on ALL data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Train the model on ALL data
    print(f"Training final {model_name} on all {len(X)} samples for deployment...")
    model.fit(X_scaled, y)
    
    # 3. Package the scaler and model together in a dictionary
    model_package = {
        'scaler': scaler,
        'model': model
    }
    
    # 4. Save to disk
    save_path = os.path.join(output_folder, groupings, f"{model_name}_production.joblib")
    os.makedirs(os.path.join(output_folder, groupings), exist_ok=True)
    joblib.dump(model_package, save_path)
    print(f"Successfully saved model package to {save_path}\n")

    print(f"Calculating feature importances for {model_name}...")
    importances = None
    method_used = ""
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        method_used = "Tree Splits"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        method_used = "Absolute Linear Coefficients"
    else:
        print("Calculating Permutation Importance (this may take a moment)...")
        result = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        method_used = "Permutation Importance"

    # 5. Plot and Save the Importance Spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, importances, color='darkred', linewidth=1.5)
    plt.fill_between(wavelengths, importances, color='red', alpha=0.3)
    
    plt.title(f"Feature Importance Spectrum\nModel: {model_name} ({method_used})")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Relative Importance")
    plt.xlim(wavelengths.min(), wavelengths.max())
    
    # Highlight top 3 peaks
    top_3_indices = np.argsort(importances)[-3:]
    for idx in top_3_indices:
        plt.axvline(x=wavelengths[idx], color='black', linestyle='--', alpha=0.5)
        plt.text(wavelengths[idx], importances[idx], f"{wavelengths[idx]:.1f}", 
                 rotation=90, va='bottom', ha='right', fontsize=9, fontweight='bold')

    plt.tight_layout()
    importance_save_path = os.path.join(IMPORTANCE_FOLDER, groupings, f"{model_name}_importance_spectrum.png")
    os.makedirs(os.path.join(IMPORTANCE_FOLDER, groupings), exist_ok=True)
    plt.savefig(importance_save_path, dpi=300)
    print(f"Saved Importance Spectrum to {importance_save_path}\n")
    plt.close()

def run_ml_pipeline(X, y, wavelengths, sample_names, groupings, n_splits=5):
    print(f"Starting ML Pipeline with {n_splits}-Fold Group Validation...\n")
    
    models = {
        "Linear SVM": SVC(kernel='linear', class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Ridge Classifier": RidgeClassifier(alpha=1.0, class_weight='balanced', random_state=42),
        "RBF SVM": SVC(kernel='rbf', class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    gkf = StratifiedGroupKFold(n_splits=n_splits)
    
    best_score = 0.0
    best_model_name = ""
    best_model = None

    for model_name, model in models.items():
        print(f"=== Training & Validating {model_name} ===")
        
        all_true = []
        all_preds = []
        
        for train_index, test_index in gkf.split(X, y, sample_names):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            scaler = StandardScaler()
            
            X_train_scaled = scaler.fit_transform(X_train)
            
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            # -----------------------------------------------
            
            all_true.extend(y_test)
            all_preds.extend(preds)
            
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(all_true, all_preds))

        current_score = f1_score(all_true, all_preds, average='weighted', zero_division=0)
        
        if current_score > best_score:
            best_score = current_score
            best_model_name = model_name
            best_model = model  # Store the actual algorithm object
        
        # --- Plotting the Confusion Matrix ---
        cm = confusion_matrix(all_true, all_preds)
        target_names = np.unique(all_true).tolist()
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_file = os.path.join(CONFUSION_MATRIX_FOLDER, groupings, f"{model_name}_confusion_matrix.png")
        os.makedirs(os.path.join(CONFUSION_MATRIX_FOLDER, groupings), exist_ok=True)
        plt.savefig(cm_file)
        print(f"Saved confusion matrix to {cm_file}")
        plt.close() # Ensure you close the plot here to save memory
        save_metrics_plot(all_true, all_preds, model_name, groupings)
    
    print(f"\nBest Model: {best_model_name} with F1-Score: {best_score:.4f}")
    save_final_model(X, y, wavelengths, best_model, best_model_name, groupings=groupings)
    
## Main Analysis Function
def analyze_data(X, y, sample_names, wavelengths, groupings="Coffee"):
    if groupings == "Breathing":
        desired_targets = ['Breathing', 'Ambient Air', 'Nitrogen']
    else:
        desired_targets = ['Blank', 'No Coffee', 'Coffee']

    mask = np.isin(y, desired_targets)
    X_filtered = X[mask]
    y_filtered = y[mask]
    sample_names_filtered = sample_names[mask]


    if groupings == "Breath":
        breath_label = ['No Coffee', 'Coffee']
        y_filtered[np.isin(y_filtered, breath_label)] = "Breath"

    print(f"Filtered dataset to {len(X_filtered)} samples with targets in {desired_targets}")

    run_pca_analysis(X_filtered, y_filtered, sample_names_filtered, groupings)
    run_tsne_analysis(X_filtered, y_filtered, sample_names_filtered, groupings)
    run_tsne_analysis(X_filtered, y_filtered, sample_names_filtered, groupings, perplexity=30)

    run_ml_pipeline(X_filtered, y_filtered, wavelengths, sample_names_filtered, groupings)


def main():
    ## Parsing for which spectra to analyze and groups
    parser = argparse.ArgumentParser(description="Animate spectra and metadata")
    parser.add_argument("--spectra", default=".", help="Spectra number to animate (default: all)")
    parser.add_argument("--groups", default=".", help="Groups to analyze (default: all)")

    args = parser.parse_args()

    if args.spectra == ".":
        spectra_numbers = range(1, 68)
    else:
        spectra_numbers = [int(num) for num in args.spectra.split(",")]

    all_X = []
    all_targets = []
    all_sample_names = []

    wavelengths = None  # Initialize wavelengths variable to be used later in importance plotting

    for num in spectra_numbers:
        metadata_file = os.path.join(METADATA_FOLDER, f"Captured_spectra_{num}_metadata.txt")
        spectra_file = os.path.join(SPECTRA_FOLDER, f"Captured_spectra_{num}.txt")
        if os.path.exists(metadata_file) and os.path.exists(spectra_file):
            wavelengths, intensities, label, group = read_spectra(spectra_file, metadata_file)

            print(f"Analyzing spectra for label: {label}, group: {group}")
            print(f"Wavelengths: {wavelengths[:5]} ...")
            print(f"Intensities shape: {intensities.shape}")
            preprocessed_spectra = preprocess_data(wavelengths, intensities, label)

            all_X.append(preprocessed_spectra)

            num_spectra = preprocessed_spectra.shape[0]
            all_targets.extend([group] * num_spectra)
            all_sample_names.extend([label] * num_spectra)

        else:
            print(f"Warning: Missing files for spectra number {num}. Skipping.")

    if len(all_X) > 0:
        X = np.vstack(all_X)
        y = np.array(all_targets)
        sample_names = np.array(all_sample_names)
        print(f"Feature Matrix (X) : {X.shape}, Labels shape: {y.shape}, Sample names shape: {sample_names.shape}")

        wavelengths = np.array(wavelengths, dtype=float)  # Ensure wavelengths are float64 for later processing
        valid_indices = wavelengths > CROP_THRESHOL
        wavelengths = wavelengths[valid_indices]
        X = X[:, valid_indices]
        analyze_data(X, y, sample_names, wavelengths, groupings="Coffee")
        analyze_data(X, y, sample_names, wavelengths, groupings="Breathing")
        analyze_data(X, y, sample_names, wavelengths, groupings="Breath")

    else:
        print("No valid spectra to analyze.")

if __name__ == "__main__":
    main()






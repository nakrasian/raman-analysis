import os
import io
import json
import zipfile
import time
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

# Import from parent directory where the scripts live
from preprocessing import SpectraPreprocessor
from prediction import SpectraClassifier

def dashboard(request):
    return render(request, 'diagnostic_app/dashboard.html')

def get_models(request):
    models_data = []
    base_path = "model/saved_models"
    if os.path.exists(base_path):
        for group in os.listdir(base_path):
            group_path = os.path.join(base_path, group)
            if os.path.isdir(group_path):
                model_files = [f for f in os.listdir(group_path) if f.endswith('.joblib')]
                models_data.append({
                    "group": group,
                    "models": model_files
                })
    return JsonResponse({"models": models_data})

def parse_file_content(content, filename):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep=None, engine='python', header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna().iloc[:, :2]
        df.columns = ['Shift', 'Intensity']
        
        chunk_size = 1024
        num_spectra = len(df) // chunk_size
        file_runs = []
        for i in range(num_spectra):
            run_df = df.iloc[i*chunk_size : (i+1)*chunk_size].copy()
            file_runs.append({
                "shift": run_df['Shift'].tolist(),
                "intensity": run_df['Intensity'].tolist()
            })
        return file_runs
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        return []

@csrf_exempt
def analyze(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
        
    try:
        use_baseline = request.POST.get('use_baseline') == 'true'
        use_snv = request.POST.get('use_snv') == 'true'
        use_minmax = request.POST.get('use_minmax') == 'true'
        
        target_group = request.POST.get('target_group')
        model_file = request.POST.get('model_file')
        
        files = request.FILES.getlist('files')
        
        response_data = {"files": {}}
        classifier = None
        
        if target_group and model_file:
            path_to_model = os.path.join("model/saved_models", target_group, model_file)
            if os.path.exists(path_to_model):
                classifier = SpectraClassifier(path_to_model)

        for f in files:
            content = f.read()
            runs = parse_file_content(content, f.name)
            
            processed_runs = []
            ml_data_list = []
            
            for run in runs:
                # ML logic dictates ALS + SNV always for prediction regardless of UI toggles
                raw_int = np.array(run["intensity"])
                ml_b = raw_int - SpectraPreprocessor.baseline_als(raw_int)
                ml_input = SpectraPreprocessor.apply_snv(ml_b)
                ml_data_list.append(ml_input)
                
                run_data = {}
                if classifier:
                    try:
                        pred, conf = classifier.predict_aggregate([ml_input])
                        run_data["prediction"] = pred
                        run_data["confidence"] = conf
                    except Exception as pred_e:
                        raise ValueError(f"predict_aggregate error on run {len(ml_data_list)} with ml_input shape {ml_input.shape}: {pred_e}")
                    
                processed_runs.append(run_data)
            
            file_result = {
                "runs": processed_runs,
                "aggregate_prediction": None,
                "aggregate_confidence": None,
                "shap_values": None,
                "shap_spec": None
            }
            
            if classifier and len(ml_data_list) > 0:
                X_data = np.array(ml_data_list)
                try:
                    agg_pred, agg_conf = classifier.predict_aggregate(X_data)
                    file_result["aggregate_prediction"] = agg_pred
                    file_result["aggregate_confidence"] = agg_conf
                    
                    shap_vals = classifier.get_shap_explanations(X_data)
                    file_result["shap_values"] = shap_vals.tolist()
                    file_result["shap_spec"] = np.mean(X_data, axis=0).tolist()
                except Exception as agg_e:
                    raise ValueError(f"predict_aggregate error on X_data shape {X_data.shape}: {agg_e}")
                
            response_data["files"][f.name] = file_result
            
        return JsonResponse(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def preprocess(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
        
    try:
        use_baseline = request.POST.get('use_baseline') == 'true'
        use_snv = request.POST.get('use_snv') == 'true'
        use_minmax = request.POST.get('use_minmax') == 'true'
        
        files = request.FILES.getlist('files')
        response_data = {"files": {}}
        
        for f in files:
            content = f.read()
            runs = parse_file_content(content, f.name)
            processed_runs = []
            
            for run in runs:
                intensities = np.array(run["intensity"])
                shift = np.array(run["shift"])
                
                baseline = None
                if use_baseline:
                    baseline = SpectraPreprocessor.baseline_als(intensities)
                    intensities = intensities - baseline
                    
                if use_snv:
                    intensities = SpectraPreprocessor.apply_snv(intensities)
                    
                if use_minmax:
                    i_min, i_max = intensities.min(), intensities.max()
                    if i_max - i_min != 0:
                        intensities = (intensities - i_min) / (i_max - i_min)
                    else:
                        intensities = intensities * 0.0
                        
                run_data = {
                    "shift": run["shift"],
                    "intensity": run["intensity"],
                    "normalized": intensities.tolist()
                }
                if baseline is not None:
                    run_data["baseline"] = baseline.tolist()
                    
                processed_runs.append(run_data)
                
            response_data["files"][f.name] = {
                "runs": processed_runs
            }
            
        return JsonResponse(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def export_report(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
        
    try:
        data = json.loads(request.POST.get('data', '{}'))
        patient_id = request.POST.get('patient_id', 'Diagnostic')
        target_group = request.POST.get('target_group', 'Unknown')
        
        memory_zip = io.BytesIO()
        with zipfile.ZipFile(memory_zip, "a", zipfile.ZIP_DEFLATED, False) as zf:
            for fname, f_data in data.items():
                runs = f_data.get('runs', [])
                if not runs:
                    continue
                    
                safe_fname = fname.replace("/", "-").replace("\\", "-").replace(".txt", "").replace(".csv", "")
                safe_fname_display = fname.replace('_', ' ')
                r_id = patient_id.replace(" ", "_").replace("/", "-")
                pdf_filename = f"{r_id}_{safe_fname}_Snapshot_Report.pdf"
                
                pdf_buffer = io.BytesIO()
                with pdf_backend.PdfPages(pdf_buffer) as pdf:
                    for frame_idx, run in enumerate(runs):
                        fig_report = plt.figure(figsize=(8.5, 11))
                        fig_report.patch.set_facecolor('#ffffff')
                        
                        agg_pred = f_data.get("aggregate_prediction", "N/A")
                        agg_conf = f_data.get("aggregate_confidence", 0)
                        loc_pred = run.get("prediction", "N/A")
                        loc_conf = run.get("confidence", 0)
                        
                        fig_report.text(0.1, 0.93, f"BREATH-CHECK Diagnostic Report", fontsize=18, fontweight='bold', color="#1e4a8d")
                        fig_report.text(0.1, 0.89, f"Patient ID: {patient_id}", fontsize=12)
                        fig_report.text(0.1, 0.86, f"Target Group: {target_group.replace('_', ' ')}", fontsize=12)
                        fig_report.text(0.1, 0.82, f"Aggregate Prediction: {agg_pred} ({agg_conf*100:.1f}%)", fontsize=14, fontweight='bold')
                        fig_report.text(0.1, 0.79, f"Frame {frame_idx+1} Snapshot: {loc_pred} ({loc_conf*100:.1f}%) | Dataset: {safe_fname_display}", fontsize=10)
                        
                        shap_importances = np.array(f_data.get("shap_values", []))
                        shift_val = np.array(run["shift"])
                        local_raw = np.array(run["intensity"])
                        
                        # Just grab ML prepped data again to plot
                        local_b = local_raw - SpectraPreprocessor.baseline_als(local_raw)
                        local_data = SpectraPreprocessor.apply_snv(local_b)
                        
                        if len(shap_importances) > 0 and not np.all(shap_importances == 0):
                            ax_s_pdf = fig_report.add_axes([0.1, 0.55, 0.8, 0.20])
                            ax_s_pdf.plot(shift_val, shap_importances, color='darkred', linewidth=1.5, label="SHAP Feature Importance")
                            ax_s_pdf.fill_between(shift_val, shap_importances, color='red', alpha=0.3)
                            ax_s_pdf.set_title("Local Interpretability Map (SHAP Impact)", fontsize=10, color="#1e4a8d")
                            ax_s_pdf.set_xlabel('RAMAN SHIFT ($cm^{-1}$)', color='#1e4a8d', fontweight='bold', fontsize=8)
                            ax_s_pdf.set_ylabel('SHAP IMPACT', color='darkred', fontweight='bold', fontsize=8)
                            ax_s_pdf.grid(True, linestyle='--', color='#d1d9e6', alpha=0.7)
                            
                            ax_sp_pdf = ax_s_pdf.twinx()
                            spec_overlay_pdf = np.array(f_data.get("shap_spec", local_data))
                            ax_sp_pdf.plot(shift_val, spec_overlay_pdf, color='#1e4a8d', linewidth=1, alpha=0.6, linestyle='-.', label="Analyzed Spectra Array")
                            ax_sp_pdf.set_ylabel('NORMALIZED INTENSITY', color='#1e4a8d', fontweight='bold', fontsize=8)
                            
                            lines_1, labels_1 = ax_s_pdf.get_legend_handles_labels()
                            lines_2, labels_2 = ax_sp_pdf.get_legend_handles_labels()
                            ax_s_pdf.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', facecolor='#ffffff', edgecolor="#d1d9e6", fontsize=7)
                            
                            for spine in ax_s_pdf.spines.values(): spine.set_color("#d1d9e6")
                            for spine in ax_sp_pdf.spines.values(): spine.set_color("#d1d9e6")
                            
                        ax_orig_pdf = fig_report.add_axes([0.1, 0.32, 0.8, 0.16])
                        ax_orig_pdf.plot(shift_val, local_raw, color='#1e4a8d', alpha=0.8, linewidth=1.5)
                        ax_orig_pdf.set_title(f"Original Spectra Snapshot (Frame {frame_idx+1})", color="#1e4a8d", fontsize=10)
                        ax_orig_pdf.grid(True, linestyle='--', color='#d1d9e6', alpha=0.7)
                        for spine in ax_orig_pdf.spines.values(): spine.set_color("#d1d9e6")
                        
                        ax_prep_pdf = fig_report.add_axes([0.1, 0.08, 0.8, 0.16])
                        ax_prep_pdf.plot(shift_val, local_data, color='#009e73', alpha=0.8, linewidth=1.5)
                        ax_prep_pdf.set_title("Preprocessed Algorithm Input (Baseline + SNV)", color="#1e4a8d", fontsize=10)
                        ax_prep_pdf.set_xlabel('RAMAN SHIFT ($cm^{-1}$)', color='#1e4a8d', fontsize=8, fontweight='bold')
                        ax_prep_pdf.grid(True, linestyle='--', color='#d1d9e6', alpha=0.7)
                        for spine in ax_prep_pdf.spines.values(): spine.set_color("#d1d9e6")
                        
                        pdf.savefig(fig_report)
                        plt.close(fig_report)
                        
                        # Only generate snapshot for the first frame to save time/space, or maybe all frames?
                        # Streamlit app seems to generate snapshot for selected frame. We'll just generate the first frame
                        # to avoid creating a massive PDF for 1000 frames. 
                        break 
                        
                zf.writestr(pdf_filename, pdf_buffer.getvalue())

        response = HttpResponse(memory_zip.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename="Diagnostic_Reports_{time.strftime("%H%M%S")}.zip"'
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import io
from io import StringIO
import joblib
from preprocessing import SpectraPreprocessor
from prediction import SpectraClassifier
import textwrap

# --- CLINICAL MONITOR THEME CONFIG ---
st.set_page_config(page_title="BREATH-CHECK | Diagnostic Interface", layout="wide")

# Light Blue Clinical Styling
st.markdown("""
    <style>
        /* Main Body Background */
        .stApp {
            background-color: #f8faff !important;
        }
        .gui-wrapper {
            border: 1px solid #d1d9e6;
            border-radius: 12px;
            padding: 20px;
            background-color: #ffffff;
        }
        .main-title { 
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
            color: #1e4a8d; 
            font-weight: 700; 
            border-bottom: 2px solid #e0e6ed; 
            padding-bottom: 15px; 
            margin-bottom: 25px;
            text-align: left;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 1.2rem;
        }
        /* Metric Cards */
        .metric-container { 
            background-color: #ffffff; 
            padding: 15px; 
            border-radius: 8px; 
            border: 1px solid #d1d9e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 10px; 
            min-height: 100px; 
        }
        .metric-label { 
            font-size: 0.75em; 
            color: #5c6c7b; 
            text-transform: uppercase; 
            font-weight: bold; 
            letter-spacing: 1px;
        }
        .metric-value { 
            font-size: 1.8em; 
            font-weight: bold; 
            color: #2c5ba9; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* --- LIGHT FILE UPLOADER --- */
        [data-testid="stFileUploader"] {
            background-color: #ffffff;
            border: 1px dashed #2c5ba9;
            border-radius: 8px;
            padding: 10px;
        }
        
        /* Input and Button Styling */
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #1e4a8d;
            border: 1px solid #d1d9e6;
        }
        hr {
            border-top: 1px solid #d1d9e6;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">BREATH CHECK | RAMAN-DIAGNOSTIC SYSTEM</div>', unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'has_run' not in st.session_state:
    st.session_state.has_run = False
if 'file_data_map' not in st.session_state:
    st.session_state.file_data_map = {}
if 'raw_file_data' not in st.session_state:
    st.session_state.raw_file_data = {}

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### Preprocessing Options")
use_baseline = st.sidebar.checkbox("Baseline Correction (ALS)", value=False)
use_snv = st.sidebar.checkbox("SNV Normalization", value=False)
use_minmax = st.sidebar.checkbox("Min-Max Normalization", value=False)

st.sidebar.markdown("### Model Classification")

# --- IDENTIFY MODELS AND OFFER DROPDOWN MENU ---
if os.path.exists("model/saved_models"):
    model_groups = [d for d in os.listdir("model/saved_models") if os.path.isdir(os.path.join("model/saved_models", d))]
else:
    model_groups = []

if model_groups:
    selected_group = st.sidebar.selectbox("Select Target Group", model_groups)
    group_path = os.path.join("model/saved_models", selected_group)
    modelOriginFiles = [f for f in os.listdir(group_path) if f.endswith(".joblib")]
    selected_model_file = st.sidebar.selectbox("Select Model", modelOriginFiles) if modelOriginFiles else None
else:
    st.sidebar.warning("No trained models found in model/saved_models directory.")
    selected_group = None
    selected_model_file = None

# --- PATIENT DASHBOARD SETTINGS ---
st.sidebar.markdown("### Patient Dashboard Settings")
show_dashboard = st.sidebar.checkbox("Show Patient Dashboard", value=True)
show_shap = st.sidebar.checkbox("Show SHAP Interpretability", value=False)

# --- DATA PROCESSING FUNCTIONS ---
def parse_file_to_runs(file_obj):
    try:
        file_obj.seek(0) 
        content = file_obj.read().decode('utf-8')
        df = pd.read_csv(StringIO(content), sep=None, engine='python', header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna().iloc[:, :2]
        df.columns = ['Shift', 'Intensity']
        
        chunk_size = 1024
        num_spectra = len(df) // chunk_size
        file_runs = []
        for i in range(num_spectra):
            run_df = df.iloc[i*chunk_size : (i+1)*chunk_size].copy()
            run_df['Normalized'] = run_df['Intensity']
            file_runs.append(run_df)
        return file_runs
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return []

def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    return text.encode('utf-8', 'ignore').decode('utf-8')

def preprocess_runs(raw_file_map, use_base, use_snv_op, use_mm_op):
    processed_map = {}
    for fname, runs in raw_file_map.items():
        proc_runs = []
        for run_df in runs:
            proc_df = run_df.copy()
            intensities = proc_df['Intensity'].values.copy()
            
            if use_base:
                baseline = SpectraPreprocessor.baseline_als(intensities)
                proc_df['Baseline'] = baseline
                intensities = intensities - baseline
            else:
                proc_df['Baseline'] = None
            
            if use_snv_op:
                intensities = SpectraPreprocessor.apply_snv(intensities)
                
            if use_mm_op:
                i_min, i_max = intensities.min(), intensities.max()
                if i_max - i_min != 0:
                    intensities = (intensities - i_min) / (i_max - i_min)
                else:
                    intensities = intensities * 0.0
                    
            proc_df['Normalized'] = intensities
            proc_runs.append(proc_df)
        processed_map[fname] = proc_runs
    return processed_map

# --- MAIN DASHBOARD CONTROLS ---
col_input1, col_input2, col_input3 = st.columns([2, 3, 1])
with col_input1:
    run_name = st.text_input("Patient / Sample ID", placeholder="ID-83746-05836")
with col_input2:
    uploaded_files = st.file_uploader("Upload Data Files", type=["txt", "csv"], accept_multiple_files=True)
with col_input3:
    st.markdown("<br>", unsafe_allow_html=True) # Vertical alignment
    if st.session_state.has_run:
        if 'report_zip' not in st.session_state:
            if st.button("PREPARE REPORTS", type="primary", use_container_width=True):
                st.session_state.trigger_pdf_export = True
        else:
            st.download_button("DOWNLOAD (.zip)", data=st.session_state.report_zip, file_name=f"Diagnostic_Reports_{time.strftime('%H%M%S')}.zip", mime="application/zip", type="primary", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

if uploaded_files:
    current_names = [f.name for f in uploaded_files]
    if list(st.session_state.raw_file_data.keys()) != current_names:
        st.session_state.raw_file_data = {f.name: parse_file_to_runs(f) for f in uploaded_files}
        st.session_state.has_run = False
        if 'report_zip' in st.session_state:
            del st.session_state.report_zip
        st.session_state.trigger_pdf_export = False
        st.rerun()

    # Apply preprocessing dynamically upon interaction
    st.session_state.file_data_map = preprocess_runs(
        st.session_state.raw_file_data, 
        use_baseline, 
        use_snv, 
        use_minmax
    )
    file_map = st.session_state.file_data_map
    
    if any(file_map.values()):
        max_seq = max((len(r) for r in file_map.values()), default=0)
        
        dashboard_container = st.container()
        st.markdown("<hr>", unsafe_allow_html=True)
        sequence_container = st.container()
        with sequence_container:
            st.markdown("### Sequence Controls")
            vis_mode = st.radio("Playback Mode:", ["Manual Slider", "Animate Sequence"], horizontal=True)
        
            ctrl_col1, ctrl_col2 = st.columns([2, 1])
        
            selected_frame = max_seq - 1
            start_trigger = False
        
            if vis_mode == "Manual Slider":
                selected_frame = ctrl_col1.slider("Spectra Index", 1, max(1, max_seq), max(1, max_seq), label_visibility="collapsed") - 1
                st.session_state.has_run = True
            else:
                btn_label = "INITIALIZE SCAN" if not st.session_state.has_run else "RERUN SEQUENCE"
                start_trigger = ctrl_col1.button(btn_label, type="primary")
                if start_trigger:
                    st.session_state.has_run = False

        # --- PREDICTION DASHBOARD (Moved above metrics) ---
        if st.session_state.has_run or start_trigger:
            if selected_group and selected_model_file:
                with dashboard_container:
                    safe_model_name = selected_model_file.split('.')[0].replace('_', ' ')
                    st.markdown("<hr>", unsafe_allow_html=True)
                
                    try:
                        path_to_model = os.path.join("model/saved_models", selected_group, selected_model_file)
                        classifier = SpectraClassifier(path_to_model)
                    
                        if show_dashboard:
                            st.markdown(f"### 📂 Patient Dashboard: **{clean_text(safe_model_name)}**")
                        
                            for fname, runs in file_map.items():
                                if len(runs) > 0:
                                    # Mandatory ML-Specific Pipeline (ALS + SNV)
                                    ml_raw = [run['Intensity'].values for run in runs]
                                    ml_base = [row - SpectraPreprocessor.baseline_als(row) for row in ml_raw]
                                    X_data = np.array(SpectraPreprocessor.apply_snv(ml_base))
                                
                                    agg_pred, agg_conf = classifier.predict_aggregate(X_data)
                                
                                    if selected_frame < len(runs):
                                        local_raw = runs[selected_frame]['Intensity'].values
                                        local_b = local_raw - SpectraPreprocessor.baseline_als(local_raw)
                                        local_data = np.array(SpectraPreprocessor.apply_snv([local_b]))
                                        local_pred, local_conf = classifier.predict_aggregate(local_data)
                                        frame_str = f"{local_pred} ({local_conf*100:.1f}%)"
                                    else:
                                        frame_str = "No Data (Finished)"
                                
                                    display_id = run_name if run_name else "Unregistered Patient"
                                    st.markdown(f"##### Patient ID: `{clean_text(display_id)}` | Source Dataset: `{clean_text(fname.replace('_', ' '))}`")
                                    db_col1, db_col2, db_col3 = st.columns(3)
                                    db_col1.metric("Target Group", clean_text(selected_group.replace('_', ' ')))
                                    db_col2.metric("Aggregate Diagnosis", f"{agg_pred} ({agg_conf*100:.1f}%)")
                                    db_col3.metric(f"Frame {selected_frame+1} Diagnosis", frame_str)
                                    
                                    if show_shap:
                                        cache_key = f"shap_{safe_model_name}_{fname}"
                                        if "shap_cache" not in st.session_state:
                                            st.session_state.shap_cache = {}
                                        
                                        if cache_key not in st.session_state.shap_cache:
                                            with st.spinner(f'Computing Aggregated SHAP Logic for {fname}...'):
                                                st.session_state.shap_cache[cache_key] = classifier.get_shap_explanations(X_data)
                                                st.session_state[f"shap_spec_{cache_key}"] = np.mean(X_data, axis=0)
                                            
                                        shap_importances = st.session_state.shap_cache[cache_key]
                                    
                                        if shap_importances is not None and not np.all(shap_importances == 0):
                                            fig_s, ax_s = plt.subplots(figsize=(12, 3))
                                            fig_s.patch.set_facecolor('#f8faff')
                                            ax_s.set_facecolor('#ffffff')
                                        
                                            shift_axis = runs[0]['Shift'].values
                                            ax_s.plot(shift_axis, shap_importances, color='darkred', linewidth=1.5, label="SHAP Feature Importance")
                                            ax_s.fill_between(shift_axis, shap_importances, color='red', alpha=0.3)
                                        
                                            ax_s.set_title("Local Interpretability Map (SHAP Impact)", fontsize=10)
                                            ax_s.set_xlabel('RAMAN SHIFT ($cm^{-1}$)', color='#1e4a8d', fontweight='bold', fontsize=8)
                                            ax_s.set_ylabel('SHAP IMPACT', color='darkred', fontweight='bold', fontsize=8)
                                            ax_s.grid(True, linestyle='--', color='#d1d9e6', alpha=0.7)
                                        
                                            ax_sp = ax_s.twinx()
                                            spec_overlay = st.session_state.get(f"shap_spec_{cache_key}", np.mean(X_data, axis=0))
                                            ax_sp.plot(shift_axis, spec_overlay, color='#1e4a8d', linewidth=1, alpha=0.6, linestyle='-.', label="Analyzed Spectra Array")
                                            ax_sp.set_ylabel('NORMALIZED INTENSITY', color='#1e4a8d', fontweight='bold', fontsize=8)
                                        
                                            lines_1, labels_1 = ax_s.get_legend_handles_labels()
                                            lines_2, labels_2 = ax_sp.get_legend_handles_labels()
                                            ax_s.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', facecolor='#ffffff', edgecolor="#d1d9e6", fontsize=7)
                                        
                                            for spine in ax_s.spines.values(): spine.set_color("#d1d9e6")
                                            for spine in ax_sp.spines.values(): spine.set_color("#d1d9e6")
                                        
                                            st.pyplot(fig_s)
                                        else:
                                            st.warning("SHAP explanation failed or empty for this model type.", icon="⚠️")
                                            fig_s = None
                                    else:
                                        fig_s = None
                                    
                            if getattr(st.session_state, 'trigger_pdf_export', False):
                                import matplotlib.backends.backend_pdf as pdf_backend
                                import zipfile
                                
                                memory_zip = io.BytesIO()
                                with st.spinner("Archiving sequential Spectral Matrices natively into memory..."):
                                    with zipfile.ZipFile(memory_zip, "a", zipfile.ZIP_DEFLATED, False) as zf:
                                        for fname, runs in file_map.items():
                                            if len(runs) > 0:
                                                ml_raw = [r['Intensity'].values for r in runs]
                                                ml_base = [row - SpectraPreprocessor.baseline_als(row) for row in ml_raw]
                                                X_data = np.array(SpectraPreprocessor.apply_snv(ml_base))
                                                cache_key = f"shap_{safe_model_name}_{fname}"
                                                
                                                safe_fname_path = fname.replace("/", "-").replace("\\", "-").replace(".txt", "").replace(".csv", "")
                                                safe_fname_display = fname.replace('_', ' ')
                                                
                                                r_id = run_name.replace(" ", "_").replace("/", "-") if run_name else "Diagnostic"
                                                pdf_filename = f"{r_id}_{safe_fname_path}_Snapshot_Report.pdf"
                                                
                                                pdf_buffer = io.BytesIO()
                                                with pdf_backend.PdfPages(pdf_buffer) as pdf:
                                                    frame_idx = min(selected_frame, len(runs) - 1)
                                                    df_run = runs[frame_idx]
                                                    local_raw = df_run['Intensity'].values
                                                    local_b = local_raw - SpectraPreprocessor.baseline_als(local_raw)
                                                    local_data = np.array(SpectraPreprocessor.apply_snv([local_b]))
                                                    local_pred, local_conf = classifier.predict_aggregate(local_data)
                                                    
                                                    fig_report = plt.figure(figsize=(8.5, 11))
                                                    fig_report.patch.set_facecolor('#ffffff')
                                                    
                                                    fig_report.text(0.1, 0.93, f"BREATH-CHECK Diagnostic Report", fontsize=18, fontweight='bold', color="#1e4a8d")
                                                    fig_report.text(0.1, 0.89, f"Patient ID: {display_id}", fontsize=12)
                                                    fig_report.text(0.1, 0.86, f"Target Group: {selected_group.replace('_', ' ')}", fontsize=12)
                                                    fig_report.text(0.1, 0.82, f"Aggregate Prediction: {agg_pred} ({agg_conf*100:.1f}%)", fontsize=14, fontweight='bold')
                                                    fig_report.text(0.1, 0.79, f"Frame {frame_idx+1} Snapshot: {local_pred} ({local_conf*100:.1f}%) | Dataset: {safe_fname_display}", fontsize=10)
                                                    
                                                    shap_importances = st.session_state.shap_cache.get(cache_key) if 'shap_cache' in st.session_state else None
                                                    if shap_importances is None:
                                                        shap_importances = classifier.get_shap_explanations(X_data)
                                                        if 'shap_cache' in st.session_state:
                                                            st.session_state.shap_cache[cache_key] = shap_importances
                                                            st.session_state[f"shap_spec_{cache_key}"] = np.mean(X_data, axis=0)

                                                    if shap_importances is not None and not np.all(shap_importances == 0):
                                                        ax_s_pdf = fig_report.add_axes([0.1, 0.55, 0.8, 0.20])
                                                        shift_axis_pdf = df_run['Shift'].values
                                                        ax_s_pdf.plot(shift_axis_pdf, shap_importances, color='darkred', linewidth=1.5, label="SHAP Feature Importance")
                                                        ax_s_pdf.fill_between(shift_axis_pdf, shap_importances, color='red', alpha=0.3)
                                                        ax_s_pdf.set_title("Local Interpretability Map (SHAP Impact)", fontsize=10, color="#1e4a8d")
                                                        ax_s_pdf.set_xlabel('RAMAN SHIFT ($cm^{-1}$)', color='#1e4a8d', fontweight='bold', fontsize=8)
                                                        ax_s_pdf.set_ylabel('SHAP IMPACT', color='darkred', fontweight='bold', fontsize=8)
                                                        ax_s_pdf.grid(True, linestyle='--', color='#d1d9e6', alpha=0.7)
                                                        
                                                        ax_sp_pdf = ax_s_pdf.twinx()
                                                        spec_overlay_pdf = st.session_state.get(f"shap_spec_{cache_key}", np.mean(X_data, axis=0))
                                                        ax_sp_pdf.plot(shift_axis_pdf, spec_overlay_pdf, color='#1e4a8d', linewidth=1, alpha=0.6, linestyle='-.', label="Analyzed Spectra Array")
                                                        ax_sp_pdf.set_ylabel('NORMALIZED INTENSITY', color='#1e4a8d', fontweight='bold', fontsize=8)
                                                        
                                                        lines_1, labels_1 = ax_s_pdf.get_legend_handles_labels()
                                                        lines_2, labels_2 = ax_sp_pdf.get_legend_handles_labels()
                                                        ax_s_pdf.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', facecolor='#ffffff', edgecolor="#d1d9e6", fontsize=7)
                                                        
                                                        for spine in ax_s_pdf.spines.values(): spine.set_color("#d1d9e6")
                                                        for spine in ax_sp_pdf.spines.values(): spine.set_color("#d1d9e6")
                                                        
                                                    ax_orig_pdf = fig_report.add_axes([0.1, 0.32, 0.8, 0.16])
                                                    shift_val = df_run['Shift'].values
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
                                                
                                                zf.writestr(pdf_filename, pdf_buffer.getvalue())

                                st.session_state.report_zip = memory_zip.getvalue()
                                st.session_state.trigger_pdf_export = False
                                st.rerun()

                        else:
                            st.markdown(f"### Classification Results: **{safe_model_name}**")
                            results_data = []
                            for fname, runs in file_map.items():
                                if selected_frame < len(runs):
                                    local_raw = runs[selected_frame]['Intensity'].values
                                    local_b = local_raw - SpectraPreprocessor.baseline_als(local_raw)
                                    local_data = np.array(SpectraPreprocessor.apply_snv([local_b]))
                                    pred, _ = classifier.predict_aggregate(local_data)
                                    results_data.append({"File": fname, "Frame": selected_frame + 1, "Prediction": pred})
                            st.dataframe(pd.DataFrame(results_data), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error initializing SpectraClassifier Dashboard: {e}")
            elif not selected_group:
                st.info("Select an ML Model from the sidebar for patient diagnostics.")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Plotting Engine Helper
        needs_second_plot = use_baseline or use_snv or use_minmax
        colors = ['#1e4a8d', '#e63946', '#2a9d8f', '#f4a261', '#9b5de5', '#00b4d8', '#f15bb5', '#38b000']

        def plot_frame(frame_idx):
            w = 8 if needs_second_plot else 12
            fig1, ax1 = plt.subplots(figsize=(w, 5))
            fig1.patch.set_facecolor('#f8faff')
            ax1.set_facecolor('#ffffff')
            
            if needs_second_plot:
                fig2, ax2 = plt.subplots(figsize=(w, 5))
                fig2.patch.set_facecolor('#f8faff')
                ax2.set_facecolor('#ffffff')
            else:
                fig2, ax2 = None, None

            all_orig = []
            all_base = []
            all_norm = []
            
            for idx, (fname, runs) in enumerate(file_map.items()):
                if frame_idx < len(runs):
                    df = runs[frame_idx]
                    c = colors[idx % len(colors)]
                    shift = df['Shift'].values
                    orig_y = df['Intensity'].values
                    all_orig.extend(orig_y)
                    
                    legend_label = textwrap.fill(fname, width=25)
                    ax1.plot(shift, orig_y, color=c, alpha=0.8, linewidth=1.5, label=legend_label)
                    
                    if use_baseline and df['Baseline'] is not None:
                        base_y = df['Baseline']
                        all_base.extend(base_y)
                        ax1.plot(shift, base_y, color=c, linestyle='--', alpha=0.6, linewidth=1.2)
                        
                    if needs_second_plot:
                        norm_y = df['Normalized'].values
                        all_norm.extend(norm_y)
                        ax2.plot(shift, norm_y, color=c, alpha=0.8, linewidth=1.5, label=legend_label)

            if len(all_orig) > 0:
                y1_min, y1_max = min(all_orig), max(all_orig)
                if all_base:
                    y1_min, y1_max = min(y1_min, min(all_base)), max(y1_max, max(all_base))
                pad = (y1_max - y1_min) * 0.05
                if pad == 0: pad = 0.5
                ax1.set_ylim(y1_min - pad, y1_max + pad)
            
            ax1.set_xlabel('RAMAN SHIFT ($cm^{-1}$)', color='#1e4a8d', fontsize=10, fontweight='bold')
            ax1.set_ylabel('RAW INTENSITY', color='#1e4a8d', fontsize=10, fontweight='bold')
            ax1.grid(True, linestyle='--', color='#d1d9e6', alpha=0.7)
            ax1.tick_params(colors='#5c6c7b')
            ax1.set_title("Original Spectra (with optional Baseline overlay)", color="#1e4a8d")
            ax1.legend(loc='upper right', facecolor='#ffffff', edgecolor="#d1d9e6", fontsize=8)
            for spine in ax1.spines.values(): spine.set_color("#d1d9e6")

            if needs_second_plot and len(all_norm) > 0:
                y2_min, y2_max = min(all_norm), max(all_norm)
                pad = (y2_max - y2_min) * 0.05
                if pad == 0: pad = 0.5
                ax2.set_ylim(y2_min - pad, y2_max + pad)
                ax2.set_xlabel('RAMAN SHIFT ($cm^{-1}$)', color='#1e4a8d', fontsize=10, fontweight='bold')
                ax2.set_ylabel('PROCESSED INTENSITY', color='#1e4a8d', fontsize=10, fontweight='bold')
                ax2.grid(True, linestyle='--', color='#d1d9e6', alpha=0.7)
                ax2.tick_params(colors='#5c6c7b')
                ax2.set_title("Preprocessed Spectra Output", color="#1e4a8d")
                ax2.legend(loc='upper right', facecolor='#ffffff', edgecolor="#d1d9e6", fontsize=8)
                for spine in ax2.spines.values(): spine.set_color("#d1d9e6")
                
            return fig1, fig2

        # Visualization Rendering
        st.markdown("### Chart Visualization")
        m_col1, m_col2 = st.columns(2)
        run_count_p = m_col1.empty()
        status_p = m_col2.empty()
        
        if needs_second_plot:
            chart_col1, chart_col2 = st.columns(2)
            chart_p1 = chart_col1.empty()
            chart_p2 = chart_col2.empty()
        else:
            chart_p1 = st.empty()
            chart_p2 = st.empty()

        if vis_mode == "Animate Sequence" and start_trigger:
            for i in range(max_seq):
                fig1, fig2 = plot_frame(i)
                active_id = run_name if run_name else "Raman_Scan"
                run_count_p.markdown(f'<div class="metric-container"><p class="metric-label">Sequence ID: {active_id}</p><p class="metric-value">{i+1} / {max_seq}</p></div>', unsafe_allow_html=True)
                status_p.markdown('<div class="metric-container"><p class="metric-label">Acquisition Status</p><p class="metric-value" style="color:#e67e22;">SCANNING...</p></div>', unsafe_allow_html=True)
                
                chart_p1.pyplot(fig1)
                plt.close(fig1)
                if fig2:
                    chart_p2.pyplot(fig2)
                    plt.close(fig2)
                time.sleep(0.08)
            
            st.session_state.has_run = True
            st.rerun()

        elif st.session_state.has_run or vis_mode == "Manual Slider":
            status_p.markdown('<div class="metric-container"><p class="metric-label">System State</p><p class="metric-value" style="color:#27ae60;">COMPLETE</p></div>', unsafe_allow_html=True)
            run_count_p.markdown(f'<div class="metric-container"><p class="metric-label">Current Frame</p><p class="metric-value">{selected_frame + 1} / {max_seq}</p></div>', unsafe_allow_html=True)
            
            fig1, fig2 = plot_frame(selected_frame)
            chart_p1.pyplot(fig1)
            plt.close(fig1)
            if fig2:
                chart_p2.pyplot(fig2)
                plt.close(fig2)

    else:
        st.info("Awaiting Data Files...")
else:
    st.info("System Ready. Please upload diagnostic data files.")

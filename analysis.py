"""
analysis.py – SERS-EBC Breathomics ML Pipeline
===============================================
Supports:
  --cv_mode loso      True Leave-One-Subject-Out CV (5 folds, one per subject)
  --cv_mode location  Original location-based grouping (legacy)
  --preprocess MODE   Preprocessing pipeline (see preprocessing.py)
  --sweep             Grid search all preprocessing modes × models, report winner
  --groups            Coffee | Breathing | Breath | all

Usage examples
--------------
# LOSO CV with default preprocessing (als_snv), Coffee group:
  python analysis.py --cv_mode loso

# LOSO CV with Adrian_Dev airPLS pipeline:
  python analysis.py --cv_mode loso --preprocess despike_airpls_snv

# Full sweep of all preprocessing × model combinations under LOSO:
  python analysis.py --cv_mode loso --sweep
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score,
                              recall_score, f1_score)

import joblib
import seaborn as sns
from preprocessing import SpectraPreprocessor


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
METADATA_FOLDER       = "metadata/4-10-2026-metadata"
SPECTRA_FOLDER        = "spectra/4-10-2026-spectra"
BASELINE_FOLDER       = "plots/baseline"
SNV_FOLDER            = "plots/snv"
PRE_PROCESSED_FOLDER  = "plots/preprocessed"
COMPONENT_ANALYSIS_FOLDER = "plots/component_analysis"
CONFUSION_MATRIX_FOLDER   = "model/confusion_matrix"
IMPORTANCE_FOLDER         = "model/feature_importance"
SPECTRA_LENGTH = 1024
CROP_THRESHOLD = 0

PREPROCESS_MODES = [
    'als_snv',
    'despike_als_snv',
    'despike_airpls_snv',
    'despike_airpls_l2',
    'despike_airpls_area',
]

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_spectra(spectra_path, metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = f.read()
    lines = metadata.splitlines()
    num_spectra = int(lines[0].split(":")[1].strip())
    label       = lines[6].split(":")[1].strip()
    group       = lines[7].split(":")[1].strip()

    df = pd.read_csv(spectra_path, header=None, names=["Wavenumber", "Intensity"])
    wavelengths = df["Wavenumber"].head(SPECTRA_LENGTH).values
    intensities = df["Intensity"].values.reshape(num_spectra, SPECTRA_LENGTH)

    print(f"  → {label} | group={group} | frames={num_spectra}")
    return wavelengths, intensities, label, group


def load_all_spectra(spectra_numbers):
    """Load raw spectra for all scan numbers. Returns (wavelengths, raw_dict).
    raw_dict maps label → {'intensities': ndarray, 'group': str}
    so we can re-preprocess without re-reading files for sweeps.
    """
    raw_dict = {}
    wavelengths = None
    for num in spectra_numbers:
        meta_path   = os.path.join(METADATA_FOLDER, f"Captured_spectra_{num}_metadata.txt")
        spectra_path = os.path.join(SPECTRA_FOLDER,  f"Captured_spectra_{num}.txt")
        if os.path.exists(meta_path) and os.path.exists(spectra_path):
            wl, intensities, label, group = read_spectra(spectra_path, meta_path)
            if wavelengths is None:
                wavelengths = wl
            raw_dict[label] = {'intensities': intensities, 'group': group}
        else:
            print(f"  Warning: missing files for scan {num}, skipping.")
    return wavelengths, raw_dict


def build_feature_matrix(raw_dict, wavelengths, preprocess_mode='als_snv',
                          group_filter=None):
    """
    Preprocess raw scans and stack into (X, y, sample_names, subject_groups).

    Parameters
    ----------
    group_filter : set or None
        If provided, only process scans whose 'group' value is in this set.
        Pass this during sweeps to skip irrelevant scans (major speed-up).

    sample_names  – full label, e.g. "Dora T6 L4"   (for location-based CV)
    subject_groups – first word of label, e.g. "Dora" (for LOSO CV)
    """
    all_X, all_y, all_sample_names, all_subject_groups = [], [], [], []

    for label, data in raw_dict.items():
        group       = data['group']
        # Skip irrelevant scans early to avoid unnecessary preprocessing
        if group_filter is not None and group not in group_filter:
            continue
        intensities = data['intensities']

        processed = SpectraPreprocessor.preprocess(intensities, mode=preprocess_mode)
        n = processed.shape[0]

        all_X.append(processed)
        all_y.extend([group] * n)
        all_sample_names.extend([label] * n)

        # Extract subject name: first token of label (e.g. "Dora", "Nathan", "Maylette")
        # For non-subject scans like "SERS substrate only L1" → "SERS" (filtered out anyway)
        subject = label.split()[0]
        all_subject_groups.extend([subject] * n)

    X            = np.vstack(all_X)
    y            = np.array(all_y)
    sample_names = np.array(all_sample_names)
    subject_grps = np.array(all_subject_groups)

    # Crop low-wavenumber noise
    wl = np.array(wavelengths, dtype=float)
    valid = wl > CROP_THRESHOLD
    return X[:, valid], y, sample_names, subject_grps, wl[valid]


# ---------------------------------------------------------------------------
# Target filtering
# ---------------------------------------------------------------------------
def desired_groups_for(groupings):
    """Return the set of raw group values needed for a given analysis grouping."""
    if groupings == 'Breathing':
        return {'Breathing', 'Ambient Air', 'Nitrogen'}
    elif groupings in ('Coffee', 'Breath'):
        return {'No Coffee', 'Coffee'}
    return None   # all groups


def filter_targets(X, y, sample_names, subject_grps, groupings):
    if groupings == "Breathing":
        desired = {'Breathing', 'Ambient Air', 'Nitrogen'}
    elif groupings in ("Coffee", "Breath"):
        desired = {'No Coffee', 'Coffee'}
    else:
        desired = set(np.unique(y))

    mask = np.isin(y, list(desired))
    X_f  = X[mask]
    y_f  = y[mask].copy()
    sn_f = sample_names[mask]
    sg_f = subject_grps[mask]

    if groupings == "Breath":
        y_f[:] = "Breath"   # collapse both coffee states into one

    print(f"  Filtered to {len(X_f)} frames | targets: {list(desired)}")
    return X_f, y_f, sn_f, sg_f


# ---------------------------------------------------------------------------
# ML pipeline
# ---------------------------------------------------------------------------
def get_models():
    return {
        "Linear SVM":       SVC(kernel='linear', class_weight='balanced', random_state=42),
        "RBF SVM":          SVC(kernel='rbf',    class_weight='balanced', random_state=42),
        "Random Forest":    RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                                   max_features='sqrt', random_state=42),
        "Ridge Classifier": RidgeClassifier(alpha=1.0, class_weight='balanced'),
        "Gradient Boosting":GradientBoostingClassifier(n_estimators=100, max_features='sqrt',
                                                       subsample=0.8, random_state=42),
    }


def run_ml_pipeline(X, y, wavelengths, group_keys, groupings, cv_mode='loso',
                    n_splits=5, preprocess_label='als_snv', save_outputs=True):
    """
    Run cross-validated ML benchmark.

    Parameters
    ----------
    group_keys : ndarray
        For cv_mode='loso'    → subject names (e.g. 'Dora', 'Nathan')
        For cv_mode='location'→ full acquisition labels (e.g. 'Dora T6 L4')
    """
    cv_label = f"LOSO ({cv_mode})" if cv_mode == 'loso' else f"Location-based ({cv_mode})"
    print(f"\n{'='*65}")
    print(f"ML Benchmark | groupings={groupings} | CV={cv_label} | preprocess={preprocess_label}")
    print(f"{'='*65}")

    unique_groups = np.unique(group_keys)
    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits < 2:
        print(f"  ⚠ Only {len(unique_groups)} group(s) — cannot cross-validate. Skipping.")
        return None, None, 0.0

    gkf = StratifiedGroupKFold(n_splits=actual_splits)
    models = get_models()

    best_f1    = 0.0
    best_name  = ""
    best_model = None

    summary_rows = []

    for model_name, model in models.items():
        print(f"\n  ── {model_name}")
        all_true, all_preds = [], []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, group_keys)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if cv_mode == 'loso':
                held_out = np.unique(group_keys[test_idx])
                print(f"    Fold {fold+1}: holding out {held_out}")

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            model.fit(X_tr_s, y_train)
            preds = model.predict(X_te_s)

            all_true.extend(y_test)
            all_preds.extend(preds)

        print(f"\n  Classification Report ({model_name}):")
        print(classification_report(all_true, all_preds))

        acc  = accuracy_score(all_true, all_preds)
        prec = precision_score(all_true, all_preds, average='weighted', zero_division=0)
        rec  = recall_score(all_true, all_preds,    average='weighted', zero_division=0)
        f1   = f1_score(all_true, all_preds,        average='weighted', zero_division=0)

        summary_rows.append({
            'model': model_name, 'preprocess': preprocess_label, 'cv_mode': cv_mode,
            'groupings': groupings, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1,
        })

        if f1 > best_f1:
            best_f1    = f1
            best_name  = model_name
            best_model = model

        if save_outputs:
            _save_confusion_matrix(all_true, all_preds, model_name,
                                   groupings, cv_mode, preprocess_label)
            _save_metrics_bar(all_true, all_preds, model_name,
                              groupings, cv_mode, preprocess_label)

    print(f"\n  ★ Best: {best_name}  F1={best_f1:.4f}")

    if save_outputs and best_model is not None:
        save_final_model(X, y, wavelengths, best_model, best_name,
                         groupings=groupings, cv_mode=cv_mode,
                         preprocess_label=preprocess_label)

    return best_name, best_model, best_f1, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Sweep: all preprocessing modes × models
# ---------------------------------------------------------------------------
def run_sweep(raw_dict, wavelengths_raw, groupings='Coffee', cv_mode='loso'):
    """
    Iterate over every preprocessing mode, build features, and run the full ML
    benchmark. Print a ranked leaderboard at the end.
    """
    print(f"\n{'#'*65}")
    print(f"SWEEP: all preprocessing modes | groupings={groupings} | CV={cv_mode}")
    print(f"{'#'*65}")

    all_summaries = []
    group_filter = desired_groups_for(groupings)

    for mode in PREPROCESS_MODES:
        print(f"\n>>> Preprocessing mode: {mode}")
        X, y, sample_names, subject_grps, wl = build_feature_matrix(
            raw_dict, wavelengths_raw, preprocess_mode=mode,
            group_filter=group_filter)

        X_f, y_f, sn_f, sg_f = filter_targets(X, y, sample_names, subject_grps, groupings)

        if len(np.unique(y_f)) < 2:
            print("  ⚠ Less than 2 classes after filtering — skipping.")
            continue

        group_keys = sg_f if cv_mode == 'loso' else sn_f

        result = run_ml_pipeline(
            X_f, y_f, wl, group_keys,
            groupings=groupings, cv_mode=cv_mode,
            n_splits=5, preprocess_label=mode,
            save_outputs=False,   # suppress per-mode file saves during sweep
        )
        if result[0] is not None:
            _, _, _, df = result
            all_summaries.append(df)

    if not all_summaries:
        print("No results to report.")
        return

    leaderboard = pd.concat(all_summaries, ignore_index=True)
    leaderboard = leaderboard.sort_values('f1', ascending=False).reset_index(drop=True)

    print(f"\n{'='*65}")
    print(f"SWEEP LEADERBOARD  (groupings={groupings}, cv={cv_mode})")
    print(f"{'='*65}")
    print(leaderboard[['preprocess','model','accuracy','precision','recall','f1']].to_string(index=False))

    best_row = leaderboard.iloc[0]
    print(f"\n★  BEST OVERALL: preprocess={best_row['preprocess']}  "
          f"model={best_row['model']}  F1={best_row['f1']:.4f}")

    # Save leaderboard CSV
    os.makedirs("model", exist_ok=True)
    lb_path = f"model/sweep_leaderboard_{groupings}_{cv_mode}.csv"
    leaderboard.to_csv(lb_path, index=False)
    print(f"   Leaderboard saved → {lb_path}")

    # Re-run the winner WITH file saves (confusion matrix, importance, saved model)
    best_mode = best_row['preprocess']
    print(f"\n>>> Re-running winner ({best_row['model']} / {best_mode}) with full output saves...")
    X, y, sample_names, subject_grps, wl = build_feature_matrix(
        raw_dict, wavelengths_raw, preprocess_mode=best_mode,
        group_filter=group_filter)
    X_f, y_f, sn_f, sg_f = filter_targets(X, y, sample_names, subject_grps, groupings)
    group_keys = sg_f if cv_mode == 'loso' else sn_f

    run_ml_pipeline(X_f, y_f, wl, group_keys,
                    groupings=groupings, cv_mode=cv_mode,
                    n_splits=5, preprocess_label=best_mode,
                    save_outputs=True)

    return leaderboard


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _cm_folder(groupings, cv_mode, preprocess_label):
    tag = f"{cv_mode}__{preprocess_label}"
    return os.path.join(CONFUSION_MATRIX_FOLDER, groupings, tag)

def _metrics_folder(groupings, cv_mode, preprocess_label):
    tag = f"{cv_mode}__{preprocess_label}"
    return os.path.join("model/metrics", groupings, tag)


def _save_confusion_matrix(all_true, all_preds, model_name, groupings, cv_mode, preprocess_label):
    folder = _cm_folder(groupings, cv_mode, preprocess_label)
    os.makedirs(folder, exist_ok=True)

    cm = confusion_matrix(all_true, all_preds)
    labels = np.unique(all_true).tolist()

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {model_name}\nCV={cv_mode} | preprocess={preprocess_label}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = os.path.join(folder, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved confusion matrix → {path}")


def _save_metrics_bar(all_true, all_preds, model_name, groupings, cv_mode, preprocess_label):
    folder = _metrics_folder(groupings, cv_mode, preprocess_label)
    os.makedirs(folder, exist_ok=True)

    acc  = accuracy_score(all_true, all_preds)
    prec = precision_score(all_true, all_preds, average='weighted', zero_division=0)
    rec  = recall_score(all_true, all_preds,    average='weighted', zero_division=0)
    f1   = f1_score(all_true, all_preds,        average='weighted', zero_division=0)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores  = [acc, prec, rec, f1]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(metrics, scores, color=['#4C72B0','#55A868','#C44E52','#8172B3'])
    plt.ylim(0, 1.15)
    plt.title(f"{model_name}\nCV={cv_mode} | preprocess={preprocess_label}")
    plt.ylabel("Score")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02,
                 f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(folder, f"{model_name}_metrics_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()


def save_final_model(X, y, wavelengths, model, model_name,
                     output_folder="model/saved_models",
                     groupings="Coffee", cv_mode='loso', preprocess_label='als_snv'):
    tag  = f"{cv_mode}__{preprocess_label}"
    folder = os.path.join(output_folder, groupings, tag)
    os.makedirs(folder, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    pkg = {'scaler': scaler, 'model': model, 'preprocess_mode': preprocess_label}
    save_path = os.path.join(folder, f"{model_name}_production.joblib")
    joblib.dump(pkg, save_path)
    print(f"  Saved production model → {save_path}")

    # Feature importance
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        method = "Tree Splits"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef[0] if coef.ndim > 1 else coef)
        method = "Absolute Coefficients"
    else:
        print("  Computing permutation importance (may take a moment)...")
        res = permutation_importance(model, X_scaled, y, n_repeats=10,
                                     random_state=42, n_jobs=-1)
        importances = res.importances_mean
        method = "Permutation Importance"

    imp_folder = os.path.join(IMPORTANCE_FOLDER, groupings, tag)
    os.makedirs(imp_folder, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, importances, color='darkred', linewidth=1.5)
    plt.fill_between(wavelengths, importances, color='red', alpha=0.3)
    plt.title(f"Feature Importance: {model_name} ({method})\nCV={cv_mode} | preprocess={preprocess_label}")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Relative Importance")
    plt.xlim(wavelengths.min(), wavelengths.max())

    top3 = np.argsort(importances)[-3:]
    for idx in top3:
        plt.axvline(x=wavelengths[idx], color='black', linestyle='--', alpha=0.5)
        plt.text(wavelengths[idx], importances[idx],
                 f"{wavelengths[idx]:.1f}", rotation=90,
                 va='bottom', ha='right', fontsize=9, fontweight='bold')

    plt.tight_layout()
    imp_path = os.path.join(imp_folder, f"{model_name}_importance_spectrum.png")
    plt.savefig(imp_path, dpi=300)
    plt.close()
    print(f"  Saved importance spectrum → {imp_path}")


# ---------------------------------------------------------------------------
# Component analysis (PCA / t-SNE)
# ---------------------------------------------------------------------------
def run_pca_analysis(X, y, sample_names, groupings, cv_mode, preprocess_label):
    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    df = pd.DataFrame({'PC1': Xp[:,0], 'PC2': Xp[:,1],
                       'Target': y, 'Sample_ID': sample_names})
    fig = px.scatter(df, x='PC1', y='PC2', color='Target', hover_name='Sample_ID',
                     title=f'PCA | {groupings} | {cv_mode} | {preprocess_label}',
                     opacity=0.7,
                     labels={'PC1': f"PC1 ({var[0]:.1f}%)",
                             'PC2': f"PC2 ({var[1]:.1f}%)"})
    out = os.path.join(COMPONENT_ANALYSIS_FOLDER, groupings,
                       f"{cv_mode}__{preprocess_label}", "pca.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out)
    print(f"  Saved PCA → {out}")


def run_tsne_analysis(X, y, sample_names, groupings, cv_mode, preprocess_label, perplexity=15):
    tsne  = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    Xt    = tsne.fit_transform(X)
    df    = pd.DataFrame({'t-SNE 1': Xt[:,0], 't-SNE 2': Xt[:,1],
                          'Target': y, 'Sample_ID': sample_names})
    fig = px.scatter(df, x='t-SNE 1', y='t-SNE 2', color='Target',
                     hover_name='Sample_ID',
                     title=f't-SNE (p={perplexity}) | {groupings} | {cv_mode} | {preprocess_label}')
    out = os.path.join(COMPONENT_ANALYSIS_FOLDER, groupings,
                       f"{cv_mode}__{preprocess_label}",
                       f"tsne_p{perplexity}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out)
    print(f"  Saved t-SNE (p={perplexity}) → {out}")


# ---------------------------------------------------------------------------
# Top-level analyze_data
# ---------------------------------------------------------------------------
def analyze_data(X, y, sample_names, subject_grps, wavelengths,
                 groupings='Coffee', cv_mode='loso', preprocess_label='als_snv'):

    X_f, y_f, sn_f, sg_f = filter_targets(X, y, sample_names, subject_grps, groupings)

    if len(np.unique(y_f)) < 2:
        print(f"  ⚠ {groupings}: fewer than 2 classes — skipping.")
        return

    group_keys = sg_f if cv_mode == 'loso' else sn_f

    run_pca_analysis(X_f, y_f, sn_f, groupings, cv_mode, preprocess_label)
    run_tsne_analysis(X_f, y_f, sn_f, groupings, cv_mode, preprocess_label, perplexity=15)
    run_tsne_analysis(X_f, y_f, sn_f, groupings, cv_mode, preprocess_label, perplexity=30)

    run_ml_pipeline(X_f, y_f, wavelengths, group_keys,
                    groupings=groupings, cv_mode=cv_mode,
                    n_splits=5, preprocess_label=preprocess_label,
                    save_outputs=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SERS-EBC Breathomics ML Pipeline")
    parser.add_argument("--spectra", default=".",
                        help="Comma-separated scan numbers, or '.' for all 1-67")
    parser.add_argument("--groups", default="Coffee",
                        choices=["Coffee", "Breathing", "Breath", "all"],
                        help="Which label grouping to analyze (default: Coffee)")
    parser.add_argument("--cv_mode", default="loso",
                        choices=["loso", "location"],
                        help="Cross-validation strategy: 'loso' (subject-level) "
                             "or 'location' (acquisition-label-level). Default: loso")
    parser.add_argument("--preprocess", default="als_snv",
                        choices=PREPROCESS_MODES,
                        help="Preprocessing pipeline. Default: als_snv")
    parser.add_argument("--sweep", action="store_true",
                        help="Grid-search all preprocessing modes × models and report winner")
    args = parser.parse_args()

    spectra_numbers = range(1, 68) if args.spectra == "." \
                      else [int(n) for n in args.spectra.split(",")]

    print(f"\nLoading spectra from {SPECTRA_FOLDER} ...")
    wavelengths_raw, raw_dict = load_all_spectra(spectra_numbers)

    if not raw_dict:
        print("No valid spectra loaded. Exiting.")
        return

    groupings_list = (["Coffee", "Breathing", "Breath"]
                      if args.groups == "all" else [args.groups])

    if args.sweep:
        for grp in groupings_list:
            run_sweep(raw_dict, wavelengths_raw, groupings=grp, cv_mode=args.cv_mode)
    else:
        print(f"\nPreprocessing with mode: {args.preprocess}")
        # When analyzing a single grouping, filter scans before preprocessing for speed
        single_filter = desired_groups_for(args.groups) if args.groups != 'all' else None
        X, y, sample_names, subject_grps, wavelengths = build_feature_matrix(
            raw_dict, wavelengths_raw, preprocess_mode=args.preprocess,
            group_filter=single_filter)

        for grp in groupings_list:
            analyze_data(X, y, sample_names, subject_grps, wavelengths,
                         groupings=grp, cv_mode=args.cv_mode,
                         preprocess_label=args.preprocess)


if __name__ == "__main__":
    main()

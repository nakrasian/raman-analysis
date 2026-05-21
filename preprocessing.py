import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, medfilt
from pybaselines.whittaker import airpls
from joblib import Parallel, delayed


class SpectraPreprocessor:
    """
    Preprocessing methods for Raman/SERS spectra.

    Pipeline options (each processes a 2D frame matrix: shape [n_frames, n_wavenumbers]):
      1. 'als_snv'            – ALS baseline correction → SNV normalization  (original)
      2. 'despike_als_snv'    – Despiking → ALS baseline → SNV
      3. 'despike_airpls_snv' – Despiking → airPLS baseline → SNV           (from Adrian_Dev)
      4. 'despike_airpls_l2'  – Despiking → airPLS baseline → L2 norm
      5. 'despike_airpls_area'– Despiking → airPLS baseline → Area norm
    """

    # ------------------------------------------------------------------
    # Baseline Correction
    # ------------------------------------------------------------------

    @staticmethod
    def baseline_als(y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares (ALS) baseline correction.
        Returns the estimated baseline vector for a single spectrum.
        """
        y = np.asarray(y, dtype=np.float64)
        if np.isnan(y).any():
            y = np.nan_to_num(y)

        L = len(y)
        data = np.ones((3, L))
        data[1] = -2 * data[1]
        diags = [0, 1, 2]
        D = sparse.spdiags(data, diags, L - 2, L)
        H = lam * D.T @ D
        w = np.ones(L)
        z = np.zeros(L)

        for _ in range(niter):
            W = sparse.diags(w, 0, shape=(L, L))
            Z = (W + H).tocsc()
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)

        return z

    @staticmethod
    def baseline_airpls(y, lam=1e6, max_iter=10):
        """
        airPLS baseline correction (from Adrian_Dev / pybaselines).
        Applies a brief Savitzky-Golay smooth before fitting to reduce noise-fitting.
        Returns the estimated baseline vector for a single spectrum.
        """
        y = np.asarray(y, dtype=np.float64)
        smoothed = savgol_filter(y, window_length=11, polyorder=3)
        baseline, _ = airpls(smoothed, lam=lam, max_iter=max_iter)
        return baseline

    # ------------------------------------------------------------------
    # Despiking  (Adrian_Dev: temporal-spatial median filter)
    # ------------------------------------------------------------------

    @staticmethod
    def despike_matrix(frame_matrix, threshold=5):
        """
        Remove cosmic-ray spikes from a frame matrix [n_frames, n_wavenumbers].
        Points that deviate > threshold × mean(|frame − median_filter(frame)|) are
        replaced with their median-filtered value.
        """
        frame_matrix = np.asarray(frame_matrix, dtype=np.float64)
        medians = medfilt(frame_matrix, kernel_size=(1, 3))
        diff = np.abs(frame_matrix - medians)
        spikes = diff > (np.mean(diff) * threshold)
        cleaned = frame_matrix.copy()
        cleaned[spikes] = medians[spikes]
        return cleaned

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def apply_snv(spectra):
        """
        Standard Normal Variate (SNV): zero-mean, unit-variance per spectrum.
        Accepts 1D (single spectrum) or 2D array [n_frames, n_wavenumbers].
        """
        spectra = np.asarray(spectra, dtype=np.float64)
        single = spectra.ndim == 1
        if single:
            spectra = spectra.reshape(1, -1)

        mean = np.mean(spectra, axis=1, keepdims=True)
        std = np.std(spectra, axis=1, keepdims=True)
        std[std == 0] = 1.0
        result = (spectra - mean) / std

        return result[0] if single else result

    @staticmethod
    def apply_l2(spectra):
        """L2 (Euclidean) normalization per spectrum."""
        spectra = np.asarray(spectra, dtype=np.float64)
        single = spectra.ndim == 1
        if single:
            spectra = spectra.reshape(1, -1)

        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result = spectra / norms
        return result[0] if single else result

    @staticmethod
    def apply_area(spectra):
        """Area (trapz integral) normalization per spectrum."""
        spectra = np.asarray(spectra, dtype=np.float64)
        single = spectra.ndim == 1
        if single:
            spectra = spectra.reshape(1, -1)

        areas = np.trapezoid(np.abs(spectra), axis=1).reshape(-1, 1)
        areas[areas == 0] = 1.0
        result = spectra / areas
        return result[0] if single else result

    # ------------------------------------------------------------------
    # High-level pipeline dispatcher
    # ------------------------------------------------------------------

    @classmethod
    def preprocess(cls, frame_matrix, mode='als_snv', airpls_lam=1e6, als_lam=1e5):
        """
        Run a complete preprocessing pipeline on a 2D frame matrix.

        Parameters
        ----------
        frame_matrix : ndarray [n_frames, n_wavenumbers]
        mode : str
            One of: 'als_snv', 'despike_als_snv', 'despike_airpls_snv',
                    'despike_airpls_l2', 'despike_airpls_area'
        airpls_lam : float
            Smoothness parameter for airPLS.
        als_lam : float
            Smoothness parameter for ALS.

        Returns
        -------
        processed : ndarray [n_frames, n_wavenumbers]
        """
        frame_matrix = np.asarray(frame_matrix, dtype=np.float64)

        # --- Optional despiking ---
        if mode.startswith('despike_'):
            frame_matrix = cls.despike_matrix(frame_matrix)

        # --- Baseline selection (parallelised across frames) ---
        use_airpls = 'airpls' in mode

        def _correct_one(spectrum):
            if use_airpls:
                baseline = cls.baseline_airpls(spectrum, lam=airpls_lam)
            else:
                baseline = cls.baseline_als(spectrum, lam=als_lam)
            return np.clip(spectrum - baseline, 0, None)

        corrected = np.array(
            Parallel(n_jobs=-1, prefer='threads')(
                delayed(_correct_one)(row) for row in frame_matrix
            )
        )

        # --- Normalization ---
        if mode.endswith('_snv'):
            return cls.apply_snv(corrected)
        elif mode.endswith('_l2'):
            return cls.apply_l2(corrected)
        elif mode.endswith('_area'):
            return cls.apply_area(corrected)
        else:
            # Fallback: SNV
            return cls.apply_snv(corrected)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class SpectraPreprocessor:
    """
    A class containing various preprocessing methods for Raman/SERS spectra.
    """

    @staticmethod
    def baseline_als(y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares baseline correction.
        Parameters:
        y : array-like
            1D array of spectra intensities.
        lam : float
            Smoothness parameter (2nd derivative penalty).
        p : float
            Asymmetry parameter.
        niter : int
            Number of iterations.
            
        Returns:
        z : array-like
            Estimated baseline.
        """
        y = np.asarray(y, dtype=np.float64)
        if np.isnan(y).any():
            y = np.nan_to_num(y)

        L = len(y)
        data = np.ones((3, L))
        data[1] = -2 * data[1]
        diags = [0, 1, 2]
        D = sparse.spdiags(data, diags, L-2, L)
        
        H = lam * D.T @ D
        w = np.ones(L)
        z = np.zeros(L)
        
        for i in range(niter):
            W = sparse.diags(w, 0, shape=(L, L))
            Z = (W + H).tocsc()
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
            
        return z

    @staticmethod
    def apply_snv(spectra):
        """
        Standard Normal Variate (SNV) transformation.
        Expects a 2D array where rows are instances (spectra) and columns are features.
        If a 1D array is passed, it temporarily converts it to 2D.
        """
        spectra = np.asarray(spectra)
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
            
        mean = np.mean(spectra, axis=1, keepdims=True)
        std = np.std(spectra, axis=1, keepdims=True)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        snv_spectra = (spectra - mean) / std
        
        if snv_spectra.shape[0] == 1:
            return snv_spectra[0]
        return snv_spectra

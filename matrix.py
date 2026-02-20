import numpy as np

class Matrix:
    # matrix initialization
    def __init__(self, m=3, n=3, min=-1, max=1, epsilon=0.1, base_P=None):
        self.m, self.n = m, n
        self.min, self.max = min, max
        self.base_P = base_P
        self.epsilon = float(epsilon)

        # Track how the current base_P was produced
        self._last_mode = None
        self._last_kwargs = {}

    def resize(self, new_m, new_n):
        self.m = int(new_m)
        self.n = int(new_n)

    # generating matrix according to params
    def generateMatrix(self, mode="uniform", **kwargs):
        """
        mode:
          - "uniform": integer matrix in [min, max]
          - "toeplitz": correlated Toeplitz-like; kwargs forwarded to generate_toeplitz
        """
        if mode == "uniform":
            self.base_P = np.random.randint(self.min, self.max + 1, size=(self.m, self.n))
            self._last_mode = "uniform"
            self._last_kwargs = {}
            return self.base_P

        if mode == "toeplitz":
            # Default rho if not provided
            rho = float(kwargs.get("rho", 0.8))
            signed = bool(kwargs.get("signed", False))
            antisymmetric = bool(kwargs.get("antisymmetric", False))
            band = kwargs.get("band", None)
            if band is not None:
                band = int(band)

            T = self.generate_toeplitz(
                rho=rho,
                signed=signed,
                antisymmetric=antisymmetric,
                band=band,
                jitter_rho=False  # initial generation uses exact rho
            )
            self.base_P = T
            self._last_mode = "toeplitz"
            # store canonical kwargs so we can perturb structurally later
            self._last_kwargs = dict(rho=rho, signed=signed, antisymmetric=antisymmetric, band=band)
            return self.base_P

        raise ValueError(f"Unknown generation mode '{mode}'. Supported: 'uniform', 'toeplitz'.")

    def returnSize(self):
        return (self.m, self.n)

    def returnEpsilon(self):
        return self.epsilon

    # add epsilon noise to matrix (structure-preserving if last mode was toeplitz)
    def generate_perturbed_matrix(self):
        """
        If the base was generated in 'toeplitz' mode, we preserve structure by
        resampling rho uniformly from [rho - epsilon, rho + epsilon], clipped to [0.0, 0.999].
        Otherwise, we add elementwise uniform noise in [-epsilon, epsilon].
        Returns a NEW Matrix instance with epsilon=0 (so it won't re-perturb on top).
        """
        if self.base_P is None:
            raise RuntimeError("Base matrix is None. Call generateMatrix(...) first.")

        if self._last_mode == "toeplitz":
            # Structured perturbation: jitter rho
            base_rho = float(self._last_kwargs.get("rho", 0.8))
            signed = bool(self._last_kwargs.get("signed", False))
            antisymmetric = bool(self._last_kwargs.get("antisymmetric", False))
            band = self._last_kwargs.get("band", None)

            # Sample rho' in [rho - eps, rho + eps], clipped
            lo = max(0.0, base_rho - self.epsilon)
            hi = min(0.999, base_rho + self.epsilon)
            rho_prime = np.random.uniform(lo, hi)

            T = self.generate_toeplitz(
                rho=rho_prime,
                signed=signed,
                antisymmetric=antisymmetric,
                band=band,
                jitter_rho=False  # we already sampled rho'
            )
            return Matrix(self.m, self.n, self.min, self.max, epsilon=0.0, base_P=T)

        # Fallback: original behavior for non-Toeplitz bases
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=(self.m, self.n))
        noise = np.round(noise, decimals=5)
        perturbed_P = self.base_P + noise
        return Matrix(self.m, self.n, self.min, self.max, epsilon=0.0, base_P=perturbed_P)

    def copy(self):
        return Matrix(self.m, self.n, self.min, self.max, self.epsilon,
                      np.copy(self.base_P) if self.base_P is not None else None)

    # ===== Toeplitz / correlated game generator =====
    def generate_toeplitz(
        self,
        rho=0.8,
        signed=False,
        antisymmetric=False,
        band=None,
        jitter_rho=True,
        clip_min=0.0,
        clip_max=0.999
    ):
        """
        Create a Toeplitz-like correlated payoff matrix of shape (m, n).

        Parameters
        ----------
        rho : float
            Base correlation decay in [0,1). Larger -> stronger correlation / lower effective rank.
        signed : bool
            If True (and antisymmetric=False), multiply by independent ±1 signs to break symmetry.
        antisymmetric : bool
            If True, produce a skew Toeplitz variant: A[i,j] = sign(j-i) * rho^{|i-j|}, diag=0.
        band : int or None
            If provided, zero-out entries where |i - j| > band (banded Toeplitz).
        jitter_rho : bool
            If True, internally jitter rho using self.epsilon (uniform in [rho-eps, rho+eps]).
            This is mostly for standalone calls; generate_perturbed_matrix handles jitter explicitly.
        clip_min, clip_max : float
            Bounds used when jittering rho.

        Returns
        -------
        np.ndarray
            Toeplitz-like matrix of shape (m, n).
        """
        rho = float(rho)
        if jitter_rho and self.epsilon > 0.0:
            lo = max(clip_min, rho - self.epsilon)
            hi = min(clip_max, rho + self.epsilon)
            rho_eff = np.random.uniform(lo, hi)
        else:
            rho_eff = rho

        m, n = self.m, self.n
        i = np.arange(m).reshape(-1, 1)
        j = np.arange(n).reshape(1, -1)
        d = np.abs(i - j)

        # Base magnitude pattern
        mag = rho_eff ** d

        if band is not None:
            if band < 0:
                raise ValueError("band must be non-negative or None.")
            mask = (d <= band).astype(float)
            mag = mag * mask

        if antisymmetric:
            # Skew Toeplitz: sign(j - i) * rho^{|i-j|}; diagonal zeroed.
            sign_part = np.sign(j - i).astype(float)
            A = sign_part * mag
            if m == n:
                np.fill_diagonal(A, 0.0)
            return A

        if signed:
            signs = np.random.choice([-1.0, 1.0], size=(m, n))
            A = mag * signs
        else:
            A = mag

        return A


class SingleCoordinateNoiseMatrix(Matrix):
    """
    Matrix variant where perturbation affects exactly one coordinate.
    If single_coordinate_noise=False, uses the base Matrix behavior.
    When single_coordinate_noise=True, a single random coordinate (i, j) is selected
    and epsilon noise is added or subtracted from that coordinate only.
    """

    def __init__(self, *args, single_coordinate_noise=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_coordinate_noise = bool(single_coordinate_noise)

    def generate_perturbed_matrix(self):
        if not self.single_coordinate_noise:
            return super().generate_perturbed_matrix()

        if self.base_P is None:
            raise RuntimeError("Base matrix is None. Call generateMatrix(...) first.")

        perturbed = self.base_P.astype(np.float64, copy=True)
        row = np.random.randint(0, self.m)
        col = np.random.randint(0, self.n)
        delta = self.epsilon if np.random.rand() >= 0.5 else -self.epsilon
        perturbed[row, col] += delta

        return SingleCoordinateNoiseMatrix(
            self.m,
            self.n,
            self.min,
            self.max,
            epsilon=0.0,
            base_P=perturbed,
            single_coordinate_noise=self.single_coordinate_noise,
        )

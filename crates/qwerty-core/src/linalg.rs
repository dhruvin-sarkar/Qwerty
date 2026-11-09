//! Dense f64 linear algebra for the classifier: shrinkage covariance and Cholesky (§1.8).

use serde::{Deserialize, Serialize};

/// Dense symmetric d×d matrix, row-major, f64. Symmetry is maintained by every mutator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymMatrix {
    dim: usize,
    data: Vec<f64>,
}

impl SymMatrix {
    pub fn zeros(dim: usize) -> Self {
        SymMatrix {
            dim,
            data: vec![0.0; dim * dim],
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.dim + j]
    }

    /// Adds `v` at (i, j) and, off the diagonal, at (j, i), so both halves stay bit-identical.
    fn add_sym(&mut self, i: usize, j: usize, v: f64) {
        let d = self.dim;
        self.data[i * d + j] += v;
        if i != j {
            self.data[j * d + i] += v;
        }
    }

    /// += w·x·xᵀ
    pub fn add_outer(&mut self, x: &[f64], w: f64) {
        debug_assert_eq!(x.len(), self.dim);
        for (i, &xi) in x.iter().enumerate().take(self.dim) {
            let wx = w * xi;
            for (j, &xj) in x.iter().enumerate().take(i + 1) {
                self.add_sym(i, j, wx * xj);
            }
        }
    }

    /// += w·other
    pub fn add_scaled(&mut self, other: &SymMatrix, w: f64) {
        debug_assert_eq!(other.dim, self.dim);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += w * b;
        }
    }

    /// += w·(a·bᵀ + b·aᵀ)
    pub fn add_cross(&mut self, a: &[f64], b: &[f64], w: f64) {
        debug_assert_eq!(a.len(), self.dim);
        debug_assert_eq!(b.len(), self.dim);
        for i in 0..self.dim {
            for j in 0..=i {
                self.add_sym(i, j, w * (a[i] * b[j] + b[i] * a[j]));
            }
        }
    }

    pub fn scale(&mut self, s: f64) {
        for a in &mut self.data {
            *a *= s;
        }
    }

    pub fn trace(&self) -> f64 {
        (0..self.dim).map(|i| self.get(i, i)).sum()
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.dim).map(|i| self.get(i, i)).collect()
    }

    /// ‖A − νI‖²_F
    pub fn frob_sq_minus_scaled_identity(&self, nu: f64) -> f64 {
        let d = self.dim;
        let mut sum = 0.0;
        for i in 0..d {
            for j in 0..d {
                let v = if i == j {
                    self.data[i * d + j] - nu
                } else {
                    self.data[i * d + j]
                };
                sum += v * v;
            }
        }
        sum
    }

    /// ‖A − B‖²_F
    pub fn frob_sq_diff(&self, other: &SymMatrix) -> f64 {
        debug_assert_eq!(other.dim, self.dim);
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// ‖A − B‖²_F over the off-diagonal entries only.
    pub fn frob_sq_offdiag_diff(&self, other: &SymMatrix) -> f64 {
        debug_assert_eq!(other.dim, self.dim);
        let d = self.dim;
        let mut sum = 0.0;
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    let v = self.data[i * d + j] - other.data[i * d + j];
                    sum += v * v;
                }
            }
        }
        sum
    }

    /// ‖A‖²_F over the off-diagonal entries only.
    fn frob_sq_offdiag(&self) -> f64 {
        let d = self.dim;
        let mut sum = 0.0;
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    let v = self.data[i * d + j];
                    sum += v * v;
                }
            }
        }
        sum
    }

    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// ν = tr(A)/d.
    fn mean_diagonal(&self) -> f64 {
        self.trace() / self.dim as f64
    }
}

/// Lower-triangular factor A + jitter·I = L·Lᵀ. `lower` is row-major packed, len d(d+1)/2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cholesky {
    pub dim: usize,
    pub lower: Vec<f64>,
    pub jitter: f64,
    pub log_det: f64,
}

/// × ν = tr(A)/d
pub const JITTER_LADDER: [f64; 5] = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2];

/// A pivot passes only when it exceeds this fraction of ν = tr(A)/d (§1.8.2, DSP-ML-17d).
const PIVOT_RELATIVE_MIN: f64 = 1e-12;
/// Floor of the `ln` argument in `log_det`. The pivot test keeps every L_ii above
/// √(1e-12·ν) > 0, so the floor never binds; it exists so no `ln` argument is unfloored (§0.8).
const LOG_DET_LN_FLOOR: f64 = f64::MIN_POSITIVE;

/// Start of row i in a row-major packed lower triangle.
fn packed_row(i: usize) -> usize {
    i * (i + 1) / 2
}

impl Cholesky {
    /// Try A first, then each ladder rung; the first rung that succeeds is recorded in `jitter` (absolute).
    pub fn factor(a: &SymMatrix) -> Result<Self, CholeskyError> {
        let dim = a.dim;
        if !a.is_finite() {
            return Err(CholeskyError::NonFinite);
        }
        let mut lower = vec![0.0; packed_row(dim)];
        if dim == 0 {
            return Ok(Cholesky {
                dim,
                lower,
                jitter: 0.0,
                log_det: 0.0,
            });
        }
        let nu = a.mean_diagonal();
        if nu.is_nan() || nu <= 0.0 {
            // tr(A) ≤ 0: A cannot be positive definite and no rung (a multiple of ν) can help.
            // Elimination fails at the first diagonal entry ≤ 0 at the latest.
            let pivot = (0..dim)
                .find(|&i| a.get(i, i) <= 0.0)
                // INVARIANT: A is finite and a sum of positive terms is positive (or +∞), so a
                // trace that is ≤ 0 or NaN (+∞ + −∞) has at least one diagonal entry ≤ 0.
                .expect("a non-positive trace has a non-positive diagonal entry");
            return Err(CholeskyError::NotPositiveDefinite { dim, pivot });
        }
        let threshold = PIVOT_RELATIVE_MIN * nu;
        let mut failed_at = match factor_into(a, 0.0, threshold, &mut lower) {
            Ok(()) => return Ok(Self::finish(dim, lower, 0.0)),
            Err(pivot) => pivot,
        };
        for rung in JITTER_LADDER {
            let jitter = rung * nu;
            match factor_into(a, jitter, threshold, &mut lower) {
                Ok(()) => return Ok(Self::finish(dim, lower, jitter)),
                Err(pivot) => failed_at = pivot,
            }
        }
        Err(CholeskyError::NotPositiveDefinite {
            dim,
            pivot: failed_at,
        })
    }

    fn finish(dim: usize, lower: Vec<f64>, jitter: f64) -> Self {
        let log_det = 2.0
            * (0..dim)
                .map(|i| lower[packed_row(i) + i].max(LOG_DET_LN_FLOOR).ln())
                .sum::<f64>();
        Cholesky {
            dim,
            lower,
            jitter,
            log_det,
        }
    }

    /// L_ij for j ≤ i.
    fn l(&self, i: usize, j: usize) -> f64 {
        self.lower[packed_row(i) + j]
    }

    /// ‖L⁻¹(x − μ)‖² by forward substitution into `scratch` (len d). No allocation.
    pub fn mahalanobis_sq(&self, x: &[f64], mean: &[f64], scratch: &mut [f64]) -> f64 {
        let d = self.dim;
        debug_assert!(x.len() >= d && mean.len() >= d && scratch.len() >= d);
        let mut total = 0.0;
        for i in 0..d {
            let row = packed_row(i);
            let mut sum = x[i] - mean[i];
            for (l, y) in self.lower[row..row + i].iter().zip(&scratch[..i]) {
                sum -= l * y;
            }
            let y = sum / self.lower[row + i];
            scratch[i] = y;
            total += y * y;
        }
        total
    }

    /// out = L⁻¹·v (forward substitution). No allocation.
    pub(crate) fn forward_solve(&self, v: &[f64], out: &mut [f64]) {
        let d = self.dim;
        debug_assert!(v.len() >= d && out.len() >= d);
        for i in 0..d {
            let row = packed_row(i);
            let mut sum = v[i];
            for (l, y) in self.lower[row..row + i].iter().zip(&out[..i]) {
                sum -= l * y;
            }
            out[i] = sum / self.lower[row + i];
        }
    }

    /// out = L⁻ᵀ·v (back substitution). No allocation.
    pub(crate) fn back_solve(&self, v: &[f64], out: &mut [f64]) {
        let d = self.dim;
        debug_assert!(v.len() >= d && out.len() >= d);
        for i in (0..d).rev() {
            let mut sum = v[i];
            for (k, y) in out.iter().enumerate().take(d).skip(i + 1) {
                sum -= self.l(k, i) * y;
            }
            out[i] = sum / self.l(i, i);
        }
    }
}

/// Row-oriented Cholesky of A + jitter·I into `lower`; Err(pivot) at the first pivot that is
/// non-finite or ≤ `threshold`.
fn factor_into(a: &SymMatrix, jitter: f64, threshold: f64, lower: &mut [f64]) -> Result<(), usize> {
    let d = a.dim;
    for i in 0..d {
        let row_i = packed_row(i);
        for j in 0..=i {
            let row_j = packed_row(j);
            let mut sum = a.data[i * d + j];
            if i == j {
                sum += jitter;
            }
            for k in 0..j {
                sum -= lower[row_i + k] * lower[row_j + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= threshold {
                    return Err(i);
                }
                lower[row_i + i] = sum.sqrt();
            } else {
                lower[row_i + j] = sum / lower[row_j + j];
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CholeskyError {
    NonFinite,
    NotPositiveDefinite { dim: usize, pivot: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Shrinkage {
    pub lambda: f64,
    pub nu: f64,
}

/// Below this target distance δ², the sample matrix already is the target: λ = 1 (§1.8.3).
const LW_MIN_TARGET_DISTANCE: f64 = 1e-12;

/// λ = 1 if δ² < 1e-12, else min(β̄², δ²)/δ².
fn lw_lambda(beta_sq: f64, delta_sq: f64) -> f64 {
    if delta_sq < LW_MIN_TARGET_DISTANCE {
        1.0
    } else {
        beta_sq.min(delta_sq) / delta_sq
    }
}

/// Grouped Ledoit-Wolf toward ν·I (§1.8.3). `mean` is S = (1/G)·Σ groups.
pub fn ledoit_wolf_identity(groups: &[SymMatrix], mean: &SymMatrix) -> Shrinkage {
    let d = mean.dim as f64;
    let nu = mean.mean_diagonal();
    if groups.is_empty() {
        // SPEC: §1.8.3 divides by G²; with no observation the estimate is all target (λ = 1).
        return Shrinkage { lambda: 1.0, nu };
    }
    let g = groups.len() as f64;
    let delta_sq = mean.frob_sq_minus_scaled_identity(nu) / d;
    let beta_sq = groups.iter().map(|y| y.frob_sq_diff(mean) / d).sum::<f64>() / (g * g);
    Shrinkage {
        lambda: lw_lambda(beta_sq, delta_sq),
        nu,
    }
}

/// Grouped Ledoit-Wolf toward the diagonal (Schäfer-Strimmer style, §1.8.3). `nu` unused (= 0).
pub fn ledoit_wolf_diagonal(groups: &[SymMatrix], mean: &SymMatrix) -> Shrinkage {
    let d = mean.dim as f64;
    if groups.is_empty() {
        // SPEC: as for the identity target, no observation means λ = 1.
        return Shrinkage {
            lambda: 1.0,
            nu: 0.0,
        };
    }
    let g = groups.len() as f64;
    let delta_sq = mean.frob_sq_offdiag() / d;
    let beta_sq = groups
        .iter()
        .map(|y| y.frob_sq_offdiag_diff(mean) / d)
        .sum::<f64>()
        / (g * g);
    Shrinkage {
        lambda: lw_lambda(beta_sq, delta_sq),
        nu: 0.0,
    }
}

/// λ·ν(s)·I + (1−λ)·s, ν(s) = tr(s)/d
pub fn shrink_to_identity(s: &SymMatrix, lambda: f64) -> SymMatrix {
    let nu = s.mean_diagonal();
    let mut out = s.clone();
    out.scale(1.0 - lambda);
    for i in 0..s.dim {
        out.data[i * s.dim + i] += lambda * nu;
    }
    out
}

/// λ·D + (1−λ)·s, D_gg = max(s_gg, DIAG_TARGET_FLOOR·ν(s))
pub fn shrink_to_diagonal(s: &SymMatrix, lambda: f64) -> SymMatrix {
    let floor = DIAG_TARGET_FLOOR * s.mean_diagonal();
    let mut out = s.clone();
    out.scale(1.0 - lambda);
    for i in 0..s.dim {
        let target = s.get(i, i).max(floor);
        out.data[i * s.dim + i] += lambda * target;
    }
    out
}

/// Floor of the diagonal target, relative to ν = tr(s)/d (DSP-ML-17b). It binds only on a dim whose
/// variance across the groups is ~0 (e.g. saturated for every negative), which would otherwise take the
/// 1e-10 jitter rung and give d²_bg ≈ 1e10·z² along it, so llr → +∞ and the background gate passes
/// anything that deviates there. Two decades below the mean variance, it leaves real dims untouched.
pub const DIAG_TARGET_FLOOR: f64 = 1e-2;

/// sorted ascending, finite, non-empty; p ∈ (0, 1]: sorted[ceil(p·n) − 1]
pub fn quantile_nearest_rank(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    debug_assert!(n > 0, "quantile of an empty list");
    debug_assert!(p > 0.0 && p <= 1.0, "quantile p outside (0, 1]");
    // The clamp is a no-op for p ∈ (0, 1]; it keeps a rounding artefact of p·n inside the list.
    let rank = ((p * n as f64).ceil() as usize).clamp(1, n);
    sorted[rank - 1]
}

/// Cyclic Jacobi stops once the off-diagonal norm is at most this fraction of the trace.
const JACOBI_REL_TOL: f64 = 1e-15;
const JACOBI_MAX_SWEEPS: usize = 60;
/// An eigenvalue at or below this fraction of the largest counts as zero (rank below `count`).
const EIGEN_RANK_REL: f64 = 1e-12;

/// Eigenvectors of Σ_r r·rᵀ for its `count` largest eigenvalues, descending, through the n×n Gram matrix [r_i·r_j]
/// (cyclic Jacobi until the off-diagonal norm ≤ 1e-15·trace, at most 60 sweeps): e = Σ_i u_i·r_i, normalized, and
/// signed so its largest-magnitude entry (lowest index on ties) is positive. An eigenvalue ≤ 1e-12 × the largest (rank
/// below `count`) gives the zero vector.
/// Training only (allocates); used by the sound map (§1.10.3 step 6).
pub fn top_eigenvectors(rows: &[Vec<f64>], count: usize) -> Vec<Vec<f64>> {
    let n = rows.len();
    let d = rows.first().map_or(0, Vec::len);
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = rows[i].iter().zip(&rows[j]).map(|(a, b)| a * b).sum();
            gram[i * n + j] = dot;
            gram[j * n + i] = dot;
        }
    }
    let (values, vectors) = jacobi_eigen(n, gram);
    let order = descending_order(&values);
    let largest = order.first().map_or(0.0, |&i| values[i]);
    (0..count)
        .map(|c| {
            let mut e = vec![0.0; d];
            let Some(&idx) = order.get(c) else {
                return e;
            };
            if largest.is_nan() || largest <= 0.0 || values[idx] <= EIGEN_RANK_REL * largest {
                return e;
            }
            for (i, row) in rows.iter().enumerate() {
                let u = vectors[i * n + idx];
                for (eg, rg) in e.iter_mut().zip(row) {
                    *eg += u * rg;
                }
            }
            normalize_signed(&mut e);
            e
        })
        .collect()
}

/// Indices of `values` sorted descending (`total_cmp`); ties keep the lower index first.
fn descending_order(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    order
}

/// Scales `e` to unit length and signs it so its largest-magnitude entry (lowest index on
/// ties) is positive. A zero vector stays zero.
fn normalize_signed(e: &mut [f64]) {
    let norm = e.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm.is_nan() || norm <= 0.0 {
        e.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let mut pivot = 0;
    for (i, v) in e.iter().enumerate() {
        if v.abs() > e[pivot].abs() {
            pivot = i;
        }
    }
    let s = if e[pivot] < 0.0 {
        -1.0 / norm
    } else {
        1.0 / norm
    };
    e.iter_mut().for_each(|v| *v *= s);
}

/// Cyclic Jacobi on the symmetric n×n row-major `a`. Returns (eigenvalues, V) with the
/// eigenvectors as the columns of the row-major n×n V, so A = V·diag(values)·Vᵀ.
fn jacobi_eigen(n: usize, mut a: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off += a[i * n + j] * a[i * n + j];
                }
            }
        }
        if off.sqrt() <= JACOBI_REL_TOL * trace.abs() {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = if theta.abs() > 1e150 {
                    0.5 / theta
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let values = (0..n).map(|i| a[i * n + i]).collect();
    (values, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng64;
    use proptest::prelude::*;

    fn random_vec(rng: &mut Rng64, d: usize) -> Vec<f64> {
        (0..d).map(|_| rng.normal()).collect()
    }

    /// BᵀB for a random (d + extra)×d Gaussian B: SPD with probability 1.
    fn random_spd(rng: &mut Rng64, d: usize, extra: usize) -> SymMatrix {
        let mut a = SymMatrix::zeros(d);
        for _ in 0..(d + extra) {
            a.add_outer(&random_vec(rng, d), 1.0);
        }
        a
    }

    fn reconstruct(c: &Cholesky) -> Vec<f64> {
        let d = c.dim;
        let mut out = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0;
                for k in 0..=i.min(j) {
                    s += c.l(i, k) * c.l(j, k);
                }
                out[i * d + j] = s;
            }
        }
        out
    }

    fn max_abs(v: &[f64]) -> f64 {
        v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
    }

    #[test]
    fn l1_cholesky_reconstructs_random_spd() {
        let mut rng = Rng64::new(0x11);
        let a = random_spd(&mut rng, 48, 12);
        let c = Cholesky::factor(&a).expect("SPD factors");
        assert_eq!(c.jitter, 0.0);
        assert_eq!(c.lower.len(), 48 * 49 / 2);
        let rec = reconstruct(&c);
        let err = rec
            .iter()
            .zip(&a.data)
            .fold(0.0f64, |m, (r, x)| m.max((r - x).abs()));
        assert!(err <= 1e-12 * max_abs(&a.data), "max|LLᵀ − A| = {err}");
        // Symmetry is exact after the mutators.
        for i in 0..48 {
            for j in 0..48 {
                assert_eq!(a.get(i, j).to_bits(), a.get(j, i).to_bits());
            }
        }
    }

    #[test]
    fn l2_cholesky_jitter_ladder_and_errors() {
        let mut rng = Rng64::new(0x12);
        let d = 12;
        // Rank d − 1 PSD.
        let mut a = SymMatrix::zeros(d);
        for _ in 0..(d - 1) {
            a.add_outer(&random_vec(&mut rng, d), 1.0);
        }
        let nu = a.trace() / d as f64;
        let c = Cholesky::factor(&a).expect("rank-deficient PSD factors with jitter");
        assert!(c.jitter >= 1e-10 * nu, "jitter {} < 1e-10·ν", c.jitter);
        assert!(c.log_det.is_finite());

        // λ_min = −1.
        let mut neg = SymMatrix::zeros(d);
        for i in 0..d {
            let mut e = vec![0.0; d];
            e[i] = 1.0;
            neg.add_outer(&e, if i == d - 1 { -1.0 } else { 1.0 });
        }
        assert!(matches!(
            Cholesky::factor(&neg),
            Err(CholeskyError::NotPositiveDefinite { dim: 12, .. })
        ));

        // NaN entry.
        let mut nan = random_spd(&mut rng, d, 4);
        nan.add_outer(&[f64::NAN; 12], 1.0);
        assert_eq!(Cholesky::factor(&nan), Err(CholeskyError::NonFinite));

        // A tiny scale changes nothing: the pivot test is relative.
        let good = random_spd(&mut rng, d, 8);
        let mut tiny = good.clone();
        tiny.scale(1e-250);
        let cg = Cholesky::factor(&good).expect("good");
        let ct = Cholesky::factor(&tiny).expect("tiny");
        assert_eq!(cg.jitter, 0.0);
        assert_eq!(ct.jitter, 0.0);
        for (lg, lt) in cg.lower.iter().zip(&ct.lower) {
            assert!((lt / 1e-125 - lg).abs() <= 1e-12 * lg.abs().max(1.0));
        }

        // Non-positive trace and the zero matrix are refused, not factored.
        assert!(matches!(
            Cholesky::factor(&SymMatrix::zeros(3)),
            Err(CholeskyError::NotPositiveDefinite { dim: 3, pivot: 0 })
        ));
    }

    #[test]
    fn l4_nearest_rank_quantile() {
        let sorted: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(quantile_nearest_rank(&sorted, 0.98), 98.0);
        assert_eq!(quantile_nearest_rank(&sorted, 0.01), 1.0);
        assert_eq!(quantile_nearest_rank(&sorted, 1.0), 100.0);
        assert_eq!(quantile_nearest_rank(&[7.0], 0.5), 7.0);
    }

    /// ln|det A| by LU with partial pivoting.
    fn lu_log_det(a: &SymMatrix) -> f64 {
        let d = a.dim;
        let mut m = a.data.clone();
        let mut log_det = 0.0;
        for col in 0..d {
            let piv = (col..d)
                .max_by(|&x, &y| m[x * d + col].abs().total_cmp(&m[y * d + col].abs()))
                .expect("non-empty range");
            if piv != col {
                for k in 0..d {
                    m.swap(col * d + k, piv * d + k);
                }
            }
            let p = m[col * d + col];
            log_det += p.abs().ln();
            for r in (col + 1)..d {
                let f = m[r * d + col] / p;
                for k in col..d {
                    m[r * d + k] -= f * m[col * d + k];
                }
            }
        }
        log_det
    }

    #[test]
    fn l5_log_det_matches_lu() {
        let mut rng = Rng64::new(0x15);
        for d in 1..=8 {
            for _ in 0..5 {
                let a = random_spd(&mut rng, d, 3);
                let c = Cholesky::factor(&a).expect("SPD");
                assert_eq!(c.jitter, 0.0);
                let lu = lu_log_det(&a);
                assert!(
                    (c.log_det - lu).abs() <= 1e-9,
                    "d={d}: {} vs {lu}",
                    c.log_det
                );
            }
        }
    }

    #[test]
    fn l6_top_eigenvectors_span_the_plane() {
        let mut rng = Rng64::new(0x16);
        let d = 48;
        let p1 = random_vec(&mut rng, d);
        let p2 = random_vec(&mut rng, d);
        let rows: Vec<Vec<f64>> = (0..10)
            .map(|_| {
                let a = 3.0 * rng.normal();
                let b = rng.normal();
                p1.iter().zip(&p2).map(|(x, y)| a * x + b * y).collect()
            })
            .collect();
        let e = top_eigenvectors(&rows, 2);
        assert_eq!(e.len(), 2);
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        assert!((dot(&e[0], &e[0]) - 1.0).abs() <= 1e-12);
        assert!((dot(&e[1], &e[1]) - 1.0).abs() <= 1e-12);
        assert!(dot(&e[0], &e[1]).abs() <= 1e-12);
        // Both plane generators lie in span(e1, e2).
        for p in [&p1, &p2] {
            let c0 = dot(p, &e[0]);
            let c1 = dot(p, &e[1]);
            let norm_sq = dot(p, p);
            assert!(((c0 * c0 + c1 * c1) - norm_sq).abs() <= 1e-10 * norm_sq);
        }
        // Equal to a dense symmetric eigensolver on Σ r·rᵀ (sign rule applied).
        let mut scatter = SymMatrix::zeros(d);
        for r in &rows {
            scatter.add_outer(r, 1.0);
        }
        let (values, vectors) = jacobi_eigen(d, scatter.data.clone());
        let order = descending_order(&values);
        for (c, ec) in e.iter().enumerate() {
            let mut dense: Vec<f64> = (0..d).map(|k| vectors[k * d + order[c]]).collect();
            normalize_signed(&mut dense);
            for (x, y) in ec.iter().zip(&dense) {
                assert!((x - y).abs() <= 1e-10, "eigenvector {c}: {x} vs {y}");
            }
        }
        // Rank 1 with count 2: the second vector is zero.
        let rank1: Vec<Vec<f64>> = (0..5)
            .map(|i| p1.iter().map(|x| x * (i as f64 + 1.0)).collect())
            .collect();
        let e = top_eigenvectors(&rank1, 2);
        assert!((dot(&e[0], &e[0]) - 1.0).abs() <= 1e-12);
        assert!(e[1].iter().all(|&v| v == 0.0));
    }

    /// Groups Y_t (mean outer product of each group's deviations from the grand variant mean)
    /// and their mean S.
    fn groups_and_mean(raw: &[Vec<Vec<f64>>], d: usize) -> (Vec<SymMatrix>, SymMatrix) {
        let g = raw.len() as f64;
        let mut m = vec![0.0; d];
        for group in raw {
            let v = group.len() as f64;
            for x in group {
                for (mi, xi) in m.iter_mut().zip(x) {
                    *mi += xi / (v * g);
                }
            }
        }
        let mut s = SymMatrix::zeros(d);
        let ys: Vec<SymMatrix> = raw
            .iter()
            .map(|group| {
                let mut y = SymMatrix::zeros(d);
                for x in group {
                    let dev: Vec<f64> = x.iter().zip(&m).map(|(a, b)| a - b).collect();
                    y.add_outer(&dev, 1.0 / group.len() as f64);
                }
                s.add_scaled(&y, 1.0 / g);
                y
            })
            .collect();
        (ys, s)
    }

    /// A λ above this is a real shrinkage, not a rounding residue of λ = 0: λ·ν clears the
    /// 1e-12·ν pivot test by two decades.
    const LAMBDA_ABOVE_ROUNDING: f64 = 1e-10;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn l3_shrinkage_properties(seed in any::<u64>(), d in 1usize..10, g in 2usize..12,
                                   v in 1usize..5, scale in -3.0f64..3.0) {
            let mut rng = Rng64::new(seed);
            let s_scale = 10f64.powf(scale);
            let raw: Vec<Vec<Vec<f64>>> = (0..g)
                .map(|_| {
                    let centre: Vec<f64> = (0..d).map(|_| rng.normal() * s_scale).collect();
                    (0..v)
                        .map(|_| centre.iter().map(|c| c + 0.5 * s_scale * rng.normal()).collect())
                        .collect()
                })
                .collect();
            let (ys, s) = groups_and_mean(&raw, d);
            let id = ledoit_wolf_identity(&ys, &s);
            let diag = ledoit_wolf_diagonal(&ys, &s);
            prop_assert!((0.0..=1.0).contains(&id.lambda));
            prop_assert!((0.0..=1.0).contains(&diag.lambda));
            prop_assert_eq!(diag.nu, 0.0);
            let shrunk = shrink_to_identity(&s, id.lambda);
            let tr = s.trace();
            prop_assert!((shrunk.trace() - tr).abs() <= 1e-12 * tr.abs());
            // SPEC: "λ > 0" is read as λ above rounding level. Two single-variant groups give
            // Y_1 = Y_2 = S exactly, so β̄² = 0 and λ = 0 in exact arithmetic, but rounding leaves a
            // λ near 1e-30 whose λ·ν·I cannot lift the rank-1 S over the 1e-12·ν pivot test.
            if id.lambda > LAMBDA_ABOVE_ROUNDING && id.nu > 0.0 {
                let c = Cholesky::factor(&shrunk).expect("identity-shrunk factors");
                prop_assert_eq!(c.jitter, 0.0);
            }
            if diag.lambda > LAMBDA_ABOVE_ROUNDING && s.trace() > 0.0 {
                let c = Cholesky::factor(&shrink_to_diagonal(&s, diag.lambda))
                    .expect("diagonal-shrunk factors");
                prop_assert_eq!(c.jitter, 0.0);
            }
        }
    }
}

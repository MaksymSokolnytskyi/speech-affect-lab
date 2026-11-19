"""Coordinate-free comparison of representations (embeddings).

The coordinates of an embedding space are arbitrary: rotate it by an orthogonal matrix Q and
the next layer can absorb Q, computing exactly the same function. Comparing two embedding
matrices element-wise is therefore meaningless; we compare them up to the action of a group
(orthogonal rotations, isotropic scaling, translation).

- `linear_cka`, `cka_matrix`: similarity of the Gram matrices ("who is close to whom");
  invariant to rotations and isotropic scaling.
- `procrustes_distance`: the smallest distance between X and YQ over all rotations Q.
- `variance_explained`: share of embedding variance explained by a grouping (eta squared).
- `class_mean_subspace`, `grassmann_distance`: the "emotion subspace" spanned by class means
  and the distance between such subspaces (points on a Grassmann manifold).
- `fisher_ratio_euclidean`, `fisher_lda`: class separability without and with the shape
  (covariance) of the within-class clouds.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import orth, subspace_angles
from sklearn.covariance import LedoitWolf


def _center(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019) between `X [n, d1]` and `Y [n, d2]` (same n items).

    CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) for centred X, Y. This equals
    HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L)) with linear kernels K = XX^T, L = YY^T, but is
    computed from d x d instead of n x n matrices.
    """
    X, Y = _center(X), _center(Y)
    xty = np.linalg.norm(Y.T @ X, "fro") ** 2
    return float(xty / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))


def cka_matrix(Xs: list[np.ndarray], Ys: list[np.ndarray]) -> np.ndarray:
    """Linear CKA between every pair `(Xs[i], Ys[j])`, as a `[len(Xs), len(Ys)]` matrix.

    Same formula as `linear_cka`, but each matrix is centred and its Gram norm computed once
    instead of once per pair. Float32 inputs stay float32 to keep memory down.
    """

    def prep(Ms: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        cent = [M - M.mean(axis=0, keepdims=True) for M in Ms]
        return cent, [float(np.linalg.norm(c.T @ c, "fro")) for c in cent]

    cx, nx = prep(Xs)
    cy, ny = (cx, nx) if Ys is Xs else prep(Ys)
    out = np.empty((len(Xs), len(Ys)))
    for i, (x, gx) in enumerate(zip(cx, nx, strict=True)):
        for j, (y, gy) in enumerate(zip(cy, ny, strict=True)):
            out[i, j] = float(np.linalg.norm(y.T @ x, "fro")) ** 2 / (gx * gy)
    return out


def procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """min_Q ||X - YQ||_F^2 over orthogonal Q, for centred X, Y scaled to unit Frobenius norm.

    Closed form: Q = UV^T from SVD(Y^T X) = U S V^T, and the minimum is 2 - 2 tr(S).
    Ranges from 0 (identical up to rotation) to 2. Dimensions must match.
    """
    X, Y = _center(X), _center(Y)
    X, Y = X / np.linalg.norm(X), Y / np.linalg.norm(Y)
    s = np.linalg.svd(Y.T @ X, compute_uv=False)
    return float(max(0.0, 2.0 - 2.0 * s.sum()))


def standardize(X: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per dimension (constant dimensions are left at zero)."""
    X = _center(X)
    sd = X.std(axis=0, keepdims=True)
    return X / np.where(sd > 0, sd, 1.0)


def variance_explained(X: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance of standardised features explained by group means (eta squared).

    Per dimension SS_between / SS_total, averaged over dimensions. Standardising first keeps
    a few huge-variance dimensions (common in transformers) from dominating.
    """
    Z = standardize(X)
    total = (Z**2).sum(axis=0)
    between = np.zeros(Z.shape[1])
    for g in np.unique(groups):
        m = groups == g
        between += m.sum() * Z[m].mean(axis=0) ** 2
    valid = total > 0
    return float(np.mean(between[valid] / total[valid]))


def _class_means(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    return classes, np.stack([Z[y == k].mean(axis=0) for k in classes])


def class_mean_subspace(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the span of class-mean deviations from the grand mean
    (standardised features); dimension <= K - 1 for K classes.

    These are "the directions along which emotions differ on average": a simple analogue of
    LDA without whitening.
    """
    _, means = _class_means(standardize(X), y)
    return orth((means - means.mean(axis=0)).T)  # [d, <= K-1]


def grassmann_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Geodesic distance on the Grassmann manifold, sqrt(sum theta_i^2), where theta are the
    principal angles between span(A) and span(B). 0 for the same subspace, at most
    pi/2 * sqrt(k)."""
    theta = subspace_angles(A, B)
    return float(np.sqrt((theta**2).sum()))


def fisher_ratio_euclidean(X: np.ndarray, y: np.ndarray) -> float:
    """Between-class over within-class sum of squares inside the class-mean subspace.

    Uses plain Euclidean distances, i.e. ignores the shape of the within-class clouds.
    """
    Z = standardize(X)
    P = Z @ class_mean_subspace(X, y)
    classes, mu = _class_means(P, y)
    grand = P.mean(axis=0)
    between = sum(
        (y == k).sum() * np.sum((m - grand) ** 2) for k, m in zip(classes, mu, strict=True)
    )
    within = sum(np.sum((P[y == k] - m) ** 2) for k, m in zip(classes, mu, strict=True))
    return float(between / within)


def fisher_lda(X: np.ndarray, y: np.ndarray) -> float:
    """Fisher-LDA criterion J = tr(S_W^{-1} S_B) on standardised features.

    S_W is the pooled within-class covariance (Ledoit-Wolf shrinkage, since d is large
    relative to the class sizes), S_B the between-class covariance of the class means.
    J measures separation in Mahalanobis rather than Euclidean geometry: directions where
    classes overlap little count more.
    """
    Z = standardize(X)
    classes, mu = _class_means(Z, y)
    idx = np.searchsorted(classes, y)
    Sw = LedoitWolf().fit(Z - mu[idx]).covariance_
    weights = np.array([(y == k).mean() for k in classes])
    Sb = (mu * weights[:, None]).T @ mu
    return float(np.trace(np.linalg.solve(Sw, Sb)))

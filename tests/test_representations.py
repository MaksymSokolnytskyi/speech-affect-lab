"""Invariances that the representation comparison relies on."""

import numpy as np
import pytest
from scipy.stats import ortho_group

from speech_affect.analysis.representations import (
    cka_matrix,
    class_mean_subspace,
    fisher_lda,
    fisher_ratio_euclidean,
    grassmann_distance,
    linear_cka,
    procrustes_distance,
    variance_explained,
)

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 20))
Q = ortho_group.rvs(20, random_state=1)


def test_cka_self_is_one():
    assert linear_cka(X, X) == pytest.approx(1.0)


def test_cka_invariant_to_rotation_scale_and_shift():
    assert linear_cka(X, 3.0 * X @ Q + 5.0) == pytest.approx(1.0)


def test_cka_low_for_independent():
    assert linear_cka(X, rng.normal(size=(300, 20))) < 0.2


def test_cka_matrix_matches_linear_cka():
    Xs = [X, X @ Q, rng.normal(size=(300, 20))]
    Ys = [rng.normal(size=(300, 7)), 2.0 * X + 1.0]
    expected = np.array([[linear_cka(a, b) for b in Ys] for a in Xs])
    np.testing.assert_allclose(cka_matrix(Xs, Ys), expected, rtol=1e-10)
    # Float32 inputs (as in exp05) agree to float32 precision.
    np.testing.assert_allclose(
        cka_matrix([x.astype(np.float32) for x in Xs], Xs), cka_matrix(Xs, Xs), atol=1e-5
    )


def test_procrustes_recovers_rotation():
    assert procrustes_distance(X, X @ Q) == pytest.approx(0.0, abs=1e-10)
    assert procrustes_distance(X, rng.normal(size=(300, 20))) > 0.5


def test_raw_difference_is_large_although_rotation_changes_nothing():
    # This is why two representations cannot simply be subtracted.
    assert np.linalg.norm(X - X @ Q) > 0.5 * np.linalg.norm(X)


def test_variance_explained_extremes():
    groups = np.repeat(np.arange(10), 30)
    means = rng.normal(size=(10, 20))
    assert variance_explained(means[groups], groups) == pytest.approx(1.0)
    assert variance_explained(X, groups) < 0.1


def test_class_mean_subspace_dimension_and_grassmann():
    y = np.repeat(np.arange(6), 50)
    Z = X + 3 * rng.normal(size=(6, 20))[y]
    A = class_mean_subspace(Z, y)
    assert A.shape == (20, 5)
    assert grassmann_distance(A, A @ ortho_group.rvs(5, random_state=2)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_fisher_criteria_grow_with_separation():
    y = np.repeat(np.arange(6), 50)
    offsets = rng.normal(size=(6, 20))
    weak, strong = X + 0.3 * offsets[y], X + 3.0 * offsets[y]
    assert fisher_ratio_euclidean(strong, y) > fisher_ratio_euclidean(weak, y)
    assert fisher_lda(strong, y) > fisher_lda(weak, y)


def test_fisher_lda_sees_what_euclidean_misses():
    # Two classes differ along a low-noise axis while another axis is dominated by
    # class-independent noise of the same scale after standardisation. LDA, which whitens by
    # the within-class covariance, separates the classes far better than plain distances.
    n = 400
    y = np.repeat([0, 1], n // 2)
    signal = np.where(y == 0, -1.0, 1.0) + 0.1 * rng.normal(size=n)
    corr_noise = rng.normal(size=n)
    Z = np.column_stack([signal + 3 * corr_noise, corr_noise])
    assert fisher_lda(Z, y) > 10 * fisher_ratio_euclidean(Z, y)

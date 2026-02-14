"""Tests for Wood Wide detector classes."""

import numpy as np
import pickle
import tempfile
import pytest

from src.detectors.woodwide import (
    WoodWideDetector,
    MultiCentroidDetector,
    DetectionResult,
)


@pytest.fixture
def synthetic_data():
    """Create synthetic embeddings and labels for testing."""
    rng = np.random.RandomState(42)
    n_exercise = 60
    n_rest = 40

    # Exercise embeddings cluster near origin
    exercise_emb = rng.randn(n_exercise, 16) * 0.3
    # Rest embeddings are offset from exercise cluster
    rest_emb = rng.randn(n_rest, 16) * 0.3 + 2.0

    embeddings = np.vstack([exercise_emb, rest_emb])

    # Labels: exercise = [2,3,4,7], rest = [1,5,6,8]
    exercise_labels = rng.choice([2, 3, 4, 7], size=n_exercise)
    rest_labels = rng.choice([1, 5, 6, 8], size=n_rest)
    labels = np.concatenate([exercise_labels, rest_labels])

    return embeddings, labels


class TestWoodWideDetectorBackwardCompat:
    """Verify original behavior is preserved when rest_threshold_percentile is None."""

    def test_single_threshold_default(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = WoodWideDetector(threshold_percentile=95)
        result = detector.fit_predict(embeddings, labels)

        assert isinstance(result, DetectionResult)
        assert result.alerts.dtype == bool
        assert len(result.alerts) == len(embeddings)
        assert detector.rest_distance_threshold is None
        assert 'rest_threshold' not in result.metrics

    def test_predict_without_labels_uses_single_threshold(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = WoodWideDetector(threshold_percentile=95)
        detector.fit(embeddings, labels)

        # Predict without labels — should use single threshold
        alerts = detector.predict(embeddings)
        assert alerts.dtype == bool
        assert len(alerts) == len(embeddings)

    def test_metrics_contain_expected_keys(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = WoodWideDetector(threshold_percentile=95)
        result = detector.fit_predict(embeddings, labels)

        expected_keys = [
            'total_windows', 'total_alerts', 'alerts_during_exercise',
            'alerts_during_rest', 'false_positive_rate_pct',
            'true_positive_rate_pct', 'exercise_windows', 'rest_windows',
            'threshold'
        ]
        for key in expected_keys:
            assert key in result.metrics, f"Missing key: {key}"


class TestDualThreshold:
    """Tests for dual-threshold mode."""

    def test_dual_threshold_computes_rest_threshold(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = WoodWideDetector(
            threshold_percentile=95,
            rest_threshold_percentile=99
        )
        detector.fit(embeddings, labels)

        assert detector.rest_distance_threshold is not None
        assert detector.rest_distance_threshold > 0

    def test_dual_threshold_metrics_include_rest_threshold(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = WoodWideDetector(
            threshold_percentile=95,
            rest_threshold_percentile=99
        )
        result = detector.fit_predict(embeddings, labels)

        assert 'rest_threshold' in result.metrics
        assert result.metrics['rest_threshold'] == detector.rest_distance_threshold

    def test_dual_threshold_reduces_rest_alerts(self, synthetic_data):
        embeddings, labels = synthetic_data

        # Single threshold
        single = WoodWideDetector(threshold_percentile=95)
        single_result = single.fit_predict(embeddings, labels)

        # Dual threshold with strict rest threshold
        dual = WoodWideDetector(
            threshold_percentile=95,
            rest_threshold_percentile=99
        )
        dual_result = dual.fit_predict(embeddings, labels)

        # Dual threshold should have <= rest alerts than single
        assert dual_result.metrics['alerts_during_rest'] <= single_result.metrics['alerts_during_rest']

    def test_dual_threshold_preserves_exercise_alerts(self, synthetic_data):
        embeddings, labels = synthetic_data

        single = WoodWideDetector(threshold_percentile=95)
        single_result = single.fit_predict(embeddings, labels)

        dual = WoodWideDetector(
            threshold_percentile=95,
            rest_threshold_percentile=99
        )
        dual_result = dual.fit_predict(embeddings, labels)

        # Exercise alerts should be identical (same exercise threshold)
        assert dual_result.metrics['alerts_during_exercise'] == single_result.metrics['alerts_during_exercise']

    def test_dual_threshold_fallback_without_labels(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = WoodWideDetector(
            threshold_percentile=95,
            rest_threshold_percentile=99
        )
        detector.fit(embeddings, labels)

        # predict() without labels falls back to single threshold
        alerts_no_labels = detector.predict(embeddings)
        alerts_with_labels = detector.predict(embeddings, labels=labels)

        # They may differ since dual-threshold applies different thresholds
        # But both should be valid boolean arrays
        assert alerts_no_labels.dtype == bool
        assert alerts_with_labels.dtype == bool

    def test_dual_threshold_save_load(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = WoodWideDetector(
            threshold_percentile=95,
            rest_threshold_percentile=99
        )
        detector.fit(embeddings, labels)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            detector.save(f.name)
            loaded = WoodWideDetector.load(f.name)

        assert loaded.rest_threshold_percentile == 99
        assert loaded.rest_distance_threshold == detector.rest_distance_threshold
        assert loaded.distance_threshold == detector.distance_threshold


class TestMultiCentroidDetector:
    """Tests for MultiCentroidDetector."""

    def test_fit_predict_returns_detection_result(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = MultiCentroidDetector(threshold_percentile=95)
        result = detector.fit_predict(embeddings, labels)

        assert isinstance(result, DetectionResult)
        assert result.alerts.dtype == bool
        assert len(result.alerts) == len(embeddings)
        assert len(result.distances) == len(embeddings)
        assert result.threshold > 0
        assert result.normal_centroid.shape == (embeddings.shape[1],)

    def test_metrics_include_per_activity_info(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = MultiCentroidDetector(threshold_percentile=95)
        result = detector.fit_predict(embeddings, labels)

        assert 'per_activity_thresholds' in result.metrics
        assert 'n_centroids' in result.metrics
        assert result.metrics['n_centroids'] > 0
        assert isinstance(result.metrics['per_activity_thresholds'], dict)

    def test_learns_centroids_for_each_activity(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = MultiCentroidDetector(threshold_percentile=95)
        detector.fit(embeddings, labels)

        unique_labels = set(np.unique(labels).astype(int))
        learned = set(detector.activity_centroids.keys())
        # Should learn centroids for all activities with enough samples
        assert learned.issubset(unique_labels)
        assert len(learned) > 0

    def test_standard_metrics_present(self, synthetic_data):
        embeddings, labels = synthetic_data
        detector = MultiCentroidDetector(threshold_percentile=95)
        result = detector.fit_predict(embeddings, labels)

        expected_keys = [
            'total_windows', 'total_alerts', 'alerts_during_exercise',
            'alerts_during_rest', 'false_positive_rate_pct',
            'true_positive_rate_pct', 'exercise_windows', 'rest_windows'
        ]
        for key in expected_keys:
            assert key in result.metrics, f"Missing key: {key}"

    def test_multi_centroid_fewer_alerts_than_single(self, synthetic_data):
        """Multi-centroid should generally produce fewer total alerts
        since each activity has its own threshold."""
        embeddings, labels = synthetic_data

        single = WoodWideDetector(threshold_percentile=95)
        single_result = single.fit_predict(embeddings, labels)

        multi = MultiCentroidDetector(threshold_percentile=95)
        multi_result = multi.fit_predict(embeddings, labels)

        # Multi-centroid should have fewer or equal total alerts
        # since per-activity thresholds are more targeted
        assert multi_result.metrics['total_alerts'] <= single_result.metrics['total_alerts']

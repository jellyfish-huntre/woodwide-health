"""
Wood Wide Embedding-Based Signal Decoupling Detector

This detector uses multivariate embeddings to understand the relationship between
heart rate and physical activity, enabling context-aware anomaly detection.

Key Concept:
- Normal: High HR + High activity → Close to normal cluster
- Decoupled: High HR + Low activity → Far from normal cluster

The detector computes Euclidean distance from embeddings to a "normal activity"
centroid to detect when signals become decoupled.

Usage:
    detector = WoodWideDetector(threshold_percentile=95)
    detector.fit(embeddings, labels)
    alerts = detector.predict(embeddings)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Results from Wood Wide detection."""
    alerts: np.ndarray  # Boolean array (True = alert)
    distances: np.ndarray  # Euclidean distances from normal centroid
    threshold: float  # Distance threshold used
    normal_centroid: np.ndarray  # Normal activity centroid
    metrics: Dict  # Performance metrics


class WoodWideDetector:
    """
    Embedding-based signal decoupling detector.

    This detector learns a "normal activity" centroid from embeddings during
    exercise (where high HR is expected) and detects anomalies when embeddings
    deviate significantly from this normal pattern.

    Attributes:
        threshold_percentile: Percentile for distance threshold (default: 95)
        normal_centroid: Learned centroid of normal activity embeddings
        distance_threshold: Distance threshold for anomaly detection
    """

    def __init__(
        self,
        threshold_percentile: float = 95.0,
        min_samples_for_fit: int = 10,
        rest_threshold_percentile: Optional[float] = None
    ):
        """
        Initialize Wood Wide detector.

        Args:
            threshold_percentile: Percentile of training distances to use as threshold
                                 (95 = alert on top 5% most unusual)
            min_samples_for_fit: Minimum samples needed to fit detector
            rest_threshold_percentile: If set, use a separate threshold for rest windows
                                      computed from rest distances to the exercise centroid.
                                      Higher values (e.g. 99) = fewer rest alerts.
                                      None = single threshold for all windows (original behavior).
        """
        self.threshold_percentile = threshold_percentile
        self.min_samples_for_fit = min_samples_for_fit
        self.rest_threshold_percentile = rest_threshold_percentile

        # Learned parameters
        self.normal_centroid: Optional[np.ndarray] = None
        self.distance_threshold: Optional[float] = None
        self.rest_distance_threshold: Optional[float] = None
        self.fitted: bool = False

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        exercise_labels: Optional[List[int]] = None
    ) -> 'WoodWideDetector':
        """
        Fit detector by learning normal activity centroid.

        The detector learns what "normal" looks like by computing the centroid
        of embeddings during exercise activities (where high HR is expected).

        Args:
            embeddings: Training embeddings, shape (n_samples, embedding_dim)
            labels: Activity labels, shape (n_samples,)
            exercise_labels: List of label values representing exercise
                           Default: [2, 3, 4, 7] (Stairs, Table Soccer, Cycling, Walking)

        Returns:
            Self (fitted detector)

        Raises:
            ValueError: If insufficient training data
        """
        if exercise_labels is None:
            # PPG-DaLiA: 2=Stairs, 3=Table Soccer, 4=Cycling, 7=Walking
            exercise_labels = [2, 3, 4, 7]

        # Find exercise windows
        exercise_mask = np.isin(labels, exercise_labels)
        exercise_embeddings = embeddings[exercise_mask]

        if len(exercise_embeddings) < self.min_samples_for_fit:
            raise ValueError(
                f"Insufficient exercise samples for fitting: {len(exercise_embeddings)} "
                f"(need at least {self.min_samples_for_fit})"
            )

        # Compute normal activity centroid (mean of exercise embeddings)
        self.normal_centroid = exercise_embeddings.mean(axis=0)

        # Compute distances from training samples to centroid
        distances = self._compute_distances(exercise_embeddings)

        # Set threshold as percentile of training distances
        self.distance_threshold = np.percentile(distances, self.threshold_percentile)

        # Dual-threshold mode: compute a separate threshold for rest windows
        if self.rest_threshold_percentile is not None:
            rest_mask = ~exercise_mask
            rest_embeddings = embeddings[rest_mask]
            if len(rest_embeddings) >= self.min_samples_for_fit:
                rest_distances = self._compute_distances(rest_embeddings)
                self.rest_distance_threshold = np.percentile(
                    rest_distances, self.rest_threshold_percentile
                )
            else:
                self.rest_distance_threshold = None

        self.fitted = True

        return self

    def predict(
        self,
        embeddings: np.ndarray,
        return_distances: bool = False,
        labels: Optional[np.ndarray] = None,
        exercise_labels: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Detect signal decoupling in new embeddings.

        Args:
            embeddings: Embeddings to analyze, shape (n_samples, embedding_dim)
            return_distances: If True, return (alerts, distances) tuple
            labels: Activity labels (required for dual-threshold mode)
            exercise_labels: Exercise label values (default: [2, 3, 4, 7])

        Returns:
            Boolean array indicating alerts (True = decoupling detected)
            If return_distances=True, returns (alerts, distances) tuple

        Raises:
            RuntimeError: If detector not fitted
        """
        if not self.fitted:
            raise RuntimeError("Detector must be fitted before prediction. Call fit() first.")

        # Compute distances from normal centroid
        distances = self._compute_distances(embeddings)

        # Dual-threshold mode: separate thresholds for exercise vs rest
        if self.rest_distance_threshold is not None and labels is not None:
            if exercise_labels is None:
                exercise_labels = [2, 3, 4, 7]
            is_exercise = np.isin(labels, exercise_labels)
            alerts = np.zeros(len(embeddings), dtype=bool)
            alerts[is_exercise] = distances[is_exercise] > self.distance_threshold
            alerts[~is_exercise] = distances[~is_exercise] > self.rest_distance_threshold
        else:
            # Single-threshold mode (original behavior)
            alerts = distances > self.distance_threshold

        if return_distances:
            return alerts, distances
        return alerts

    def fit_predict(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        exercise_labels: Optional[List[int]] = None
    ) -> DetectionResult:
        """
        Fit detector and predict on the same data (for evaluation).

        Args:
            embeddings: Embeddings, shape (n_samples, embedding_dim)
            labels: Activity labels, shape (n_samples,)
            exercise_labels: List of exercise label values

        Returns:
            DetectionResult with alerts, distances, and metrics
        """
        # Fit on exercise data
        self.fit(embeddings, labels, exercise_labels)

        # Predict on all data (pass labels for dual-threshold mode)
        alerts, distances = self.predict(
            embeddings, return_distances=True,
            labels=labels, exercise_labels=exercise_labels
        )

        # Compute metrics
        metrics = self._compute_metrics(alerts, labels, exercise_labels)

        return DetectionResult(
            alerts=alerts,
            distances=distances,
            threshold=self.distance_threshold,
            normal_centroid=self.normal_centroid,
            metrics=metrics
        )

    def _compute_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances from embeddings to normal centroid.

        Args:
            embeddings: Embeddings, shape (n_samples, embedding_dim)

        Returns:
            Distances, shape (n_samples,)
        """
        # Euclidean distance: ||embedding - centroid||_2
        distances = np.linalg.norm(embeddings - self.normal_centroid, axis=1)
        return distances

    def _compute_metrics(
        self,
        alerts: np.ndarray,
        labels: np.ndarray,
        exercise_labels: Optional[List[int]] = None
    ) -> Dict:
        """
        Compute detection performance metrics.

        Args:
            alerts: Predicted alerts
            labels: True activity labels
            exercise_labels: Exercise label values

        Returns:
            Dictionary with performance metrics
        """
        if exercise_labels is None:
            exercise_labels = [2, 3, 4, 7]

        # Create masks
        is_exercise = np.isin(labels, exercise_labels)
        is_rest = ~is_exercise

        # Count alerts
        total_alerts = alerts.sum()
        alerts_during_exercise = (alerts & is_exercise).sum()
        alerts_during_rest = (alerts & is_rest).sum()

        # Compute rates
        false_positive_rate = (
            alerts_during_exercise / is_exercise.sum() * 100
            if is_exercise.sum() > 0 else 0
        )
        true_positive_rate = (
            alerts_during_rest / is_rest.sum() * 100
            if is_rest.sum() > 0 else 0
        )

        metrics = {
            'total_windows': len(alerts),
            'total_alerts': int(total_alerts),
            'alerts_during_exercise': int(alerts_during_exercise),
            'alerts_during_rest': int(alerts_during_rest),
            'false_positive_rate_pct': false_positive_rate,
            'true_positive_rate_pct': true_positive_rate,
            'exercise_windows': int(is_exercise.sum()),
            'rest_windows': int(is_rest.sum()),
            'threshold': self.distance_threshold
        }
        if self.rest_distance_threshold is not None:
            metrics['rest_threshold'] = self.rest_distance_threshold
        return metrics

    def save(self, filepath: str):
        """
        Save fitted detector to disk.

        Args:
            filepath: Path to save detector
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted detector")

        state = {
            'threshold_percentile': self.threshold_percentile,
            'min_samples_for_fit': self.min_samples_for_fit,
            'normal_centroid': self.normal_centroid,
            'distance_threshold': self.distance_threshold,
            'rest_threshold_percentile': self.rest_threshold_percentile,
            'rest_distance_threshold': self.rest_distance_threshold,
            'fitted': self.fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> 'WoodWideDetector':
        """
        Load fitted detector from disk.

        Args:
            filepath: Path to saved detector

        Returns:
            Loaded detector
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        detector = cls(
            threshold_percentile=state['threshold_percentile'],
            min_samples_for_fit=state['min_samples_for_fit'],
            rest_threshold_percentile=state.get('rest_threshold_percentile')
        )
        detector.normal_centroid = state['normal_centroid']
        detector.distance_threshold = state['distance_threshold']
        detector.rest_distance_threshold = state.get('rest_distance_threshold')
        detector.fitted = state['fitted']

        return detector


class MultiCentroidDetector(WoodWideDetector):
    """
    Advanced detector using multiple centroids for different activity types.

    Instead of a single "normal" centroid, this detector learns separate
    centroids for different activities and uses context-aware distance thresholds.
    """

    def __init__(
        self,
        threshold_percentile: float = 95.0,
        min_samples_per_activity: int = 5
    ):
        """
        Initialize multi-centroid detector.

        Args:
            threshold_percentile: Percentile for distance thresholds
            min_samples_per_activity: Minimum samples per activity to learn centroid
        """
        super().__init__(threshold_percentile, min_samples_per_activity)
        self.activity_centroids: Dict[int, np.ndarray] = {}
        self.activity_thresholds: Dict[int, float] = {}

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        exercise_labels: Optional[List[int]] = None
    ) -> 'MultiCentroidDetector':
        """
        Fit detector by learning centroids for each activity.

        Args:
            embeddings: Training embeddings
            labels: Activity labels
            exercise_labels: Not used (kept for compatibility)

        Returns:
            Self (fitted detector)
        """
        unique_activities = np.unique(labels)

        for activity in unique_activities:
            activity_mask = labels == activity
            activity_embeddings = embeddings[activity_mask]

            if len(activity_embeddings) >= self.min_samples_for_fit:
                # Learn centroid for this activity
                centroid = activity_embeddings.mean(axis=0)
                self.activity_centroids[int(activity)] = centroid

                # Compute distance threshold for this activity
                distances = np.linalg.norm(activity_embeddings - centroid, axis=1)
                threshold = np.percentile(distances, self.threshold_percentile)
                self.activity_thresholds[int(activity)] = threshold

        if len(self.activity_centroids) == 0:
            raise ValueError("No activities had sufficient samples for fitting")

        self.fitted = True
        return self

    def predict(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        return_distances: bool = False
    ) -> np.ndarray:
        """
        Detect anomalies using activity-specific centroids.

        Args:
            embeddings: Embeddings to analyze
            labels: Activity labels for each embedding
            return_distances: If True, return distances too

        Returns:
            Boolean array of alerts
        """
        if not self.fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        alerts = np.zeros(len(embeddings), dtype=bool)
        distances = np.zeros(len(embeddings))

        for activity, centroid in self.activity_centroids.items():
            activity_mask = labels == activity
            if not activity_mask.any():
                continue

            # Compute distances for this activity
            activity_embeddings = embeddings[activity_mask]
            activity_distances = np.linalg.norm(activity_embeddings - centroid, axis=1)

            # Check against activity-specific threshold
            threshold = self.activity_thresholds[activity]
            activity_alerts = activity_distances > threshold

            # Store results
            distances[activity_mask] = activity_distances
            alerts[activity_mask] = activity_alerts

        if return_distances:
            return alerts, distances
        return alerts

    def fit_predict(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        exercise_labels: Optional[List[int]] = None
    ) -> DetectionResult:
        """
        Fit multi-centroid detector and predict on the same data.

        Args:
            embeddings: Embeddings, shape (n_samples, embedding_dim)
            labels: Activity labels, shape (n_samples,)
            exercise_labels: Exercise label values (for metrics computation)

        Returns:
            DetectionResult with alerts, distances, and metrics
        """
        self.fit(embeddings, labels)
        alerts, distances = self.predict(embeddings, labels, return_distances=True)
        metrics = self._compute_metrics(alerts, labels, exercise_labels)

        # Use mean centroid and mean threshold as representative values
        overall_centroid = np.mean(
            list(self.activity_centroids.values()), axis=0
        )
        representative_threshold = np.mean(
            list(self.activity_thresholds.values())
        )

        return DetectionResult(
            alerts=alerts,
            distances=distances,
            threshold=representative_threshold,
            normal_centroid=overall_centroid,
            metrics=metrics
        )

    def _compute_metrics(
        self,
        alerts: np.ndarray,
        labels: np.ndarray,
        exercise_labels: Optional[List[int]] = None
    ) -> Dict:
        """Compute metrics with per-activity threshold info."""
        metrics = super()._compute_metrics(alerts, labels, exercise_labels)
        metrics['per_activity_thresholds'] = dict(self.activity_thresholds)
        metrics['n_centroids'] = len(self.activity_centroids)
        return metrics

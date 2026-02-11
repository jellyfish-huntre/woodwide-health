"""
Isolation Forest Baseline Detector for Health Monitoring.

This serves as a more sophisticated baseline compared to simple HR thresholding.
It uses scikit-learn's Isolation Forest algorithm on raw features (PPG, ACC_X, ACC_Y, ACC_Z, ACC_MAG)
to detect anomalies without understanding the contextual relationship between signals.

Key Difference from Wood Wide:
- Isolation Forest: Detects unusual feature combinations (statistical outliers)
- Wood Wide: Understands signal relationships (e.g., high HR during exercise is normal)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, NamedTuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestResult(NamedTuple):
    """Results from Isolation Forest detection."""
    alerts: np.ndarray  # Boolean array of anomaly alerts
    anomaly_scores: np.ndarray  # Raw anomaly scores from model
    threshold: float  # Detection threshold used


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector for multivariate health data.

    This detector uses the Isolation Forest algorithm to identify unusual
    patterns in the raw feature space (PPG + accelerometer). Unlike Wood Wide's
    embedding-based approach, it doesn't understand contextual relationships
    and will often flag exercise as anomalous.

    Example:
        >>> detector = IsolationForestDetector(contamination=0.1)
        >>> detector.fit(training_windows, training_labels)
        >>> result = detector.predict(test_windows)
        >>> print(f"Anomalies detected: {result.alerts.sum()}")
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            contamination: Expected proportion of anomalies (0.1 = 10%)
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        # Initialize model and scaler
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.scaler = StandardScaler()

        # State
        self.is_fitted = False
        self.feature_names = ['PPG', 'ACC_X', 'ACC_Y', 'ACC_Z', 'ACC_MAG']

    def _extract_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from windows.

        Uses mean and std of each signal over the window as features.
        This gives the model 10 features: mean and std for each of 5 signals.

        Args:
            windows: Shape (n_windows, window_length, n_features)

        Returns:
            Feature matrix of shape (n_windows, n_features * 2)
        """
        n_windows = windows.shape[0]

        # Compute mean and std for each signal
        means = windows.mean(axis=1)  # (n_windows, 5)
        stds = windows.std(axis=1)    # (n_windows, 5)

        # Concatenate into feature vector
        features = np.concatenate([means, stds], axis=1)  # (n_windows, 10)

        return features

    def fit(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        exercise_labels: list = [2, 3, 4, 5]
    ) -> 'IsolationForestDetector':
        """
        Fit detector on training data.

        Like Wood Wide, we train on exercise data (where high HR is expected)
        to learn what "normal activity" looks like. The difference is that
        Isolation Forest only sees raw features, not learned relationships.

        Args:
            windows: Training windows of shape (n_windows, window_length, 5)
            labels: Activity labels for each window
            exercise_labels: Which labels represent exercise (default: cycling, walking, stairs)

        Returns:
            self (for method chaining)
        """
        # Filter to exercise windows (same as Wood Wide)
        exercise_mask = np.isin(labels, exercise_labels)
        exercise_windows = windows[exercise_mask]

        if len(exercise_windows) < 10:
            raise ValueError(
                f"Insufficient exercise samples for fitting: {len(exercise_windows)} "
                f"(need at least 10)"
            )

        print(f"Isolation Forest: Training on {len(exercise_windows)} exercise windows")

        # Extract features
        features = self._extract_features(exercise_windows)

        # Fit scaler
        features_scaled = self.scaler.fit_transform(features)

        # Fit Isolation Forest
        self.model.fit(features_scaled)

        self.is_fitted = True

        return self

    def predict(self, windows: np.ndarray) -> IsolationForestResult:
        """
        Predict anomalies on new data.

        Args:
            windows: Windows to analyze, shape (n_windows, window_length, 5)

        Returns:
            IsolationForestResult with alerts and scores

        Raises:
            RuntimeError: If detector hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction. Call fit() first.")

        # Extract and scale features
        features = self._extract_features(windows)
        features_scaled = self.scaler.transform(features)

        # Get predictions (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(features_scaled)
        alerts = (predictions == -1)

        # Get anomaly scores (more negative = more anomalous)
        scores = self.model.score_samples(features_scaled)

        return IsolationForestResult(
            alerts=alerts,
            anomaly_scores=scores,
            threshold=self.model.offset_  # Decision threshold learned during fit
        )

    def save(self, filepath: str):
        """Save fitted detector to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted detector")

        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, filepath: str) -> 'IsolationForestDetector':
        """Load fitted detector from disk."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        detector = cls(
            contamination=save_data['contamination'],
            n_estimators=save_data['n_estimators'],
            random_state=save_data['random_state']
        )

        detector.model = save_data['model']
        detector.scaler = save_data['scaler']
        detector.is_fitted = save_data['is_fitted']
        detector.feature_names = save_data['feature_names']

        return detector


def analyze_performance(
    result: IsolationForestResult,
    labels: np.ndarray,
    activity_map: dict = None
) -> dict:
    """
    Analyze detector performance by activity.

    Args:
        result: Detection result
        labels: Activity labels for each window
        activity_map: Optional mapping from label to activity name

    Returns:
        Performance metrics dictionary
    """
    if activity_map is None:
        activity_map = {
            1: 'Sitting',
            2: 'Cycling',
            3: 'Walking',
            4: 'Ascending stairs',
            5: 'Descending stairs'
        }

    # Overall metrics
    total_alerts = result.alerts.sum()
    alert_rate = result.alerts.mean()

    # Exercise vs rest
    exercise_labels = [2, 3, 4, 5]
    exercise_mask = np.isin(labels, exercise_labels)
    rest_mask = ~exercise_mask

    exercise_alerts = (result.alerts & exercise_mask).sum()
    rest_alerts = (result.alerts & rest_mask).sum()

    exercise_total = exercise_mask.sum()
    rest_total = rest_mask.sum()

    # By activity
    by_activity = {}
    for label, activity in activity_map.items():
        mask = labels == label
        if mask.sum() > 0:
            n_alerts = (result.alerts & mask).sum()
            n_total = mask.sum()
            by_activity[activity] = {
                'alerts': int(n_alerts),
                'total': int(n_total),
                'rate': float(n_alerts / n_total) if n_total > 0 else 0.0
            }

    return {
        'total_windows': len(labels),
        'total_alerts': int(total_alerts),
        'alert_rate': float(alert_rate),
        'exercise': {
            'alerts': int(exercise_alerts),
            'total': int(exercise_total),
            'rate': float(exercise_alerts / exercise_total) if exercise_total > 0 else 0.0
        },
        'rest': {
            'alerts': int(rest_alerts),
            'total': int(rest_total),
            'rate': float(rest_alerts / rest_total) if rest_total > 0 else 0.0
        },
        'by_activity': by_activity
    }

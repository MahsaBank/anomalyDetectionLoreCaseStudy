# detector.py
from collections import deque
from statistics import mean
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.svm import OneClassSVM

class SentimentWindow:
    def __init__(self, window_size=10, buffer_size=15):
        self.raw_window = deque(maxlen=10)
        self.combined_window = deque(maxlen=10)
        self.training_buffer = []
        self.buffer_size = buffer_size
        self.iforest_model = IsolationForest(contamination=0.1, random_state=42)
        self.svm_model = OneClassSVM(kernel='rbf', nu=0.1, gamma="auto")
        self.trained = False

    def update(self, raw_score, combined_score):
        self.raw_window.append(raw_score)
        self.combined_window.append(combined_score)
        self.training_buffer.append(combined_score)

        if len(self.training_buffer) >= self.buffer_size:
            self.train_model()

    def train_model(self):
        # X = np.array(self.training_buffer).reshape(-1, 1)
        if len(self.training_buffer) < 2:
            return
        deltas = np.diff(self.training_buffer).reshape(-1, 1)
        self.iforest_model.fit(deltas)
        self.svm_model.fit(deltas)
        self.trained = True

    def is_anomalous(self, threshold=0.5):
        """
        Returns:
            result (bool): True if anomaly detected by either method.
            details (dict): Which methods triggered the anomaly.
        """
        if len(self.combined_window) < 2:
            return False, {"rule_based": False, "ml_based_iforest": False, "ml_based_svm": False}

        prev = self.combined_window[-2]
        curr = self.combined_window[-1]
        # Rule-based delta check
        delta = abs(curr - prev)
        # delta = abs(self.combined_window[-1] - mean(list(self.combined_window)[:-1]))
        rule_based = delta > threshold

        # ML-based check
        if self.trained:
            # current_point = np.array([[self.combined_window[-1]]])
            # ml_based = self.model.predict(current_point)[0] == -1
            delta_array = np.array([[curr - prev]])
            ml_iforest = self.iforest_model.predict(delta_array)[0] == -1
            ml_svm = self.svm_model.predict(delta_array)[0] == -1
        else:
            ml_iforest = ml_svm = False

        return rule_based or ml_iforest or ml_svm, {
            "rule_based": rule_based,
            "ml_based_iforest": ml_iforest,
            "ml_based_svm": ml_svm
        }
    
    def get_averages(self):
        return {
            "raw_avg": mean(self.raw_window) if self.raw_window else 0,
            "combined_avg": mean(self.combined_window) if self.combined_window else 0
        }

import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump, load


class HallucinationDetector:
    """
    Simple logistic regression classifier for hallucination probability.
    """

    def __init__(self, feature_keys):
        self.feature_keys = feature_keys
        self.model = LogisticRegression()

    def featurize_batch(self, feature_dicts):
        """
        Convert list of feature dicts → 2D array for classifier
        """
        X = []
        for f in feature_dicts:
            X.append([f[k] for k in self.feature_keys])
        return np.array(X)

    def train(self, feature_dicts, labels):
        """
        labels: 0 = non-hallucination, 1 = hallucination
        """
        X = self.featurize_batch(feature_dicts)
        y = np.array(labels)
        self.model.fit(X, y)

    def predict_proba(self, feature_dict):
        X = self.featurize_batch([feature_dict])
        return float(self.model.predict_proba(X)[0, 1])

    def save(self, path):
        dump({"model": self.model, "keys": self.feature_keys}, path)

    @staticmethod
    def load(path):
        data = load(path)
        det = HallucinationDetector(data["keys"])
        det.model = data["model"]
        return det

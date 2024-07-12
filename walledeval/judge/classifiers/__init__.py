# walledeval/judge/classifiers.py

import json
import numpy as np

from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from onnxruntime import InferenceSession

import numpy.typing as npt

__all__ = [
    "Ridge", "XGBoost", "OnnxInference"
]


class Ridge(RidgeClassifier):
    @classmethod
    def from_config(cls, config_path: str, **kwargs):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        # assert "attributes" in config, "Key 'attributes' missing in config"
        
        attributes = config.pop("attributes", {})
        
        classifier = cls(**kwargs)
        classifier.set_params(**config)
        
        if "coef_" in attributes:
            classifier.coef_ = np.array(attributes["coef_"])
        if "intercept_" in attributes:
            classifier.intercept_ = np.array(attributes["intercept_"])
        if "n_features_in_" in attributes:
            classifier.n_features_in_ = np.array(attributes["n_features_in_"])
        
        return classifier
    
    def predict(self, input: npt.ArrayLike) -> list[float]:
        decision = self.decision_function(input)
        
        decision = np.c_[-decision, decision]
        probs = np.exp(decision) / np.sum(np.exp(decision))
        preds = probs[:, 1]
        return preds.tolist()


class XGBoost(XGBClassifier):
    @classmethod
    def from_config(cls, config_path: str, **kwargs):
        classifier = cls(**kwargs)
        classifier.load_model(config_path)
        return classifier
        
    def predict(self, input: npt.ArrayLike) -> list[float]:
        preds = self.predict_proba(input)[:, 1]
        return preds.tolist()
    
class OnnxInference(InferenceSession):
    def predict(self, input: npt.ArrayLike) -> list[float]:
        input_name = self.get_inputs()[0].name
        X_input = np.array(input, dtype=np.float32)
        
        preds = self.run(None, {input_name: X_input})
        probs = preds[1]
        probs_harmful = [prob[1] for prob in probs]
        return probs_harmful
    
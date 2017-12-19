from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import typing
from builtins import zip
import os
import io
from future.utils import PY3
from typing import Any, Optional
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

import numpy as np
import sklearn
from sklearn.svm import OneClassSVM

class SklearnOneClassSvmIntentClassifier(Component):
    name = "sklearn_one_class_svm_intent_classifier"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    def __init__(self, clf=None):
        # type: (List[sklearn.svm.OneClassSVM]) -> None
        self.clf = clf

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy", "sklearn"]

    def train(self, training_data, config, **kwargs):
        labels = [e.get("intent") for e in training_data.intent_examples]
        classes = list(set(labels))

        if len(classes) < 2:
            raise RuntimeError("Can not train an intent classifier. Need at least 2 different classes. ")
        else:
            X = np.stack([example.get("text_features") for example in training_data.intent_examples])
            self.clf = {}
            for row in classes:
                self.clf[row] = OneClassSVM(
                    kernel="linear",
                    nu=0.1,
                    gamma=0.5
                )
            for label in classes:
                instances = X[[i for i, row in enumerate(labels) if row == label]]
                self.clf[label].fit(instances)

    def process(self, message, **kwargs):
        if not self.clf:
            raise RuntimeError("Can not find proper classifiers")
        else:
            X = message.get("text_features").reshape(1, -1)
            # score of decision function is positive for an inlier and negative for an outlier
            result = [{
                "name": label,
                "confidence": self.clf[label].decision_function(X)[0][0]
            } for label in self.clf.keys()]
            result = sorted(result, key=lambda x: x["confidence"], reverse=True)

            message.set("intent", result[0], add_to_output=True)
            message.set("intent_ranking", result, add_to_output=True)

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> SklearnIntentClassifier
        import cloudpickle

        if model_dir and model_metadata.get("sklearn_one_class_svm_intent_classifier"):
            classifier_file = os.path.join(model_dir, model_metadata.get("sklearn_one_class_svm_intent_classifier"))
            with io.open(classifier_file, 'rb') as f:  # pragma: no test
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return SklearnOneClassSvmIntentClassifier()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        import cloudpickle

        classifier_file = os.path.join(model_dir, "sklearn_one_class_svm_intent_classifier.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "sklearn_one_class_svm_intent_classifier": "sklearn_one_class_svm_intent_classifier.pkl"
        }

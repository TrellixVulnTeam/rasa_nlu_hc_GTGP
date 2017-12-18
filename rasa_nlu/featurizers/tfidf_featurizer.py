from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import typing
import io
import os
import sys
import cloudpickle
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.tokenizers import Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

from sklearn.feature_extraction.text import TfidfVectorizer

# if typing.TYPE_CHECKING:
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     import numpy as np
#     from builtins import str

class TfidfFeaturizer(Featurizer):
    name = "tfidf_featurizer"

    provides = ["text_features"]

    requires = ["tokens"]

    def __init__(self):
        self.vectorizer = None

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["sklearn", "numpy"]

    def train(self, training_data, config, **kwargs):
        self.vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "))

        sentences = [" ".join([token.text for token in tokens.get("tokens")]) for tokens in
                     training_data.intent_examples]
        print("training tf idf vectorizer")
        data = self.vectorizer.fit_transform(sentences).toarray()

        for i, example in enumerate(training_data.intent_examples):
            example.set("text_features", self._combine_with_existing_text_features(example, data[i]))

    def process(self, message, **kwargs):
        sentence = " ".join([token.text for token in message.get("tokens")])

        result = self.vectorizer.transform([sentence])
        result = result.toarray()[0]
        message.set("text_features", self._combine_with_existing_text_features(message, result))

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        if model_dir and model_metadata.get("tfidf_featurizer"):
            classifier_file = os.path.join(model_dir, model_metadata.get("tfidf_featurizer"))
            with io.open(classifier_file, 'rb') as f:  # pragma: no test
                if sys.version_info[0] == 3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return TfidfFeaturizer()

    def persist(self, model_dir):
        if self.vectorizer:
            vectorizer_file = os.path.join(model_dir, "tfidf_featurizer.pkl")
            with io.open(vectorizer_file, 'wb') as f:
                cloudpickle.dump(self, f)
            return {
                "tfidf_featurizer": "tfidf_featurizer.pkl"
            }
        else:
            raise RuntimeError("tfidf vectorizer is not initialized!")

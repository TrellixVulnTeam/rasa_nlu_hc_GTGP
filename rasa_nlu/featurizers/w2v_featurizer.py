from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import typing
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

import numpy as np
import pickle

class W2vfeaturizer(Featurizer):
    name = "w2v_featurizer"

    provides = ["text_features"]

    requires = ["tokens"]

    with open("./data/w2v.pickle", "rb") as file:
        w2v_dict = pickle.load(file)
    for key in w2v_dict.keys():
        w2v_dict[key] = np.array(w2v_dict[key])
    dim = list(w2v_dict.values())[0].shape[0]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        for example in training_data.intent_examples:
            features = self.sum_up_features(example.get("tokens"))
            example.set("text_features", self._combine_with_existing_text_features(example, features))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        features = self.sum_up_features(message.get("tokens"))
        message.set("text_features", self._combine_with_existing_text_features(message, features))

    def sum_up_features(self, tokens):
        vec = np.zeros(self.dim)
        for token in tokens:
            try:
                vec += self.w2v_dict[token.text]
            except KeyError:
                pass
        if tokens:
            vec /= len(tokens)

        return vec

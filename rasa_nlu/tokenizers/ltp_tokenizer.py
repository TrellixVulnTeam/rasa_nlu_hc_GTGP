from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

from pyltp import Segmentor

class LTPTokenizer(Tokenizer, Component):

    name = "tokenizer_ltp"

    provides = ["tokens"]

    def __init__(self):
        # 本地加载ltp分词器
        self.seg = Segmentor()
        self.seg.load("/mnt/hgfs/vmware_share_folder/ltp_data/cws.model")

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["pyltp"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        if config['language'] != 'zh':
            raise Exception("tokenizer_ltp is only used for Chinese. Check your configure json file.")
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]

        tokens = []
        start = 0
        for word in self.seg.segment(text):
            tokens.append(Token(word, start))
            start += len(word)

        return tokens

if __name__ == "__main__":
    ltp_tokenizer = LTPTokenizer()
    tokens = ltp_tokenizer.tokenize("")
    print()

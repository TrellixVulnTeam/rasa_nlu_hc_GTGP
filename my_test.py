from rasa_nlu.model import Metadata, Interpreter, Trainer
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.converters import load_data

# training
training_data = load_data("./data/examples/rasa/demo-rasa_zh.json")
trainer = Trainer(RasaNLUConfig("./sample_configs/_my_config.json"))
trainer.train(training_data)
model_directory = trainer.persist("./models/")
print("saved model directory: %s" % model_directory)

# prediction
# interpreter = Interpreter.load("./models/default/model_20171211-152650",
#                                RasaNLUConfig("./sample_configs/_my_config.json"))
# while True:
#     inputs = input("Q: ")
#     result = interpreter.parse(inputs)
#     print(result)

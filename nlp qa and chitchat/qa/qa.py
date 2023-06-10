import os
import re
from typing import Union
from simpletransformers.question_answering import QuestionAnsweringModel
# from pythainlp.util import normalize, maiyamok


# def preprocess(text):
#     text = text.replace("\t", " ")
#     text = text.replace("\n", " ")
#     text = normalize(text)
#     text = re.sub("\s+", " ", text)
#     return text.strip()
from pythainlp.util import normalize, maiyamok
import re

def preprocess_text(text):
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = normalize(text)
    text = re.sub("\s+", " ", text)
    return text.strip()

class SimpleTransformer:
    def __init__(self,
                 model_type: str, 
                 model_name: Union[str, os.PathLike], 
                 cuda_device: int=-1):
        
        self.model = QuestionAnsweringModel(model_type, 
                                            model_name, 
                                            cuda_device=cuda_device)
    
    
    def __str__(self):
        return self.model_type, self.model_name, self.cuda_device
    
    
    def answer(self, question: str, context: str):
        to_predict = self.convert2input(question, context)
        answers, probabilities = self.model.predict(to_predict)
        # return answers, probabilities
        # print(answers[0]["answer"][0], len(answers))
        # print(probabilities)
        return answers[0]["answer"][0]
        
        
    def convert2input(self, question: str, context: str):
        return [{
            "context": context,
            "qas": [{
                "question": question,
                "id": "0"
            }]
        }]
        
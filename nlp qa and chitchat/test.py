# from transformers import pipeline
# import torch
# from pythainlp.util import normalize, maiyamok
# import re

# def preprocess_text(text):
#     text = text.replace("\t", " ")
#     text = text.replace("\n", " ")
#     text = normalize(text)
#     text = re.sub("\s+", " ", text)
#     return text.strip()

# QA_THRESH = 1e-4
# question_answerer = pipeline("question-answering", 
#                              model="parinzee/mdeberta-cmsk-qa", 
#                              device=torch.device("cuda:1"))

# question = preprocess_text("ใครเป็นผู้ออกตราสารหนี้ภาคเอกชน ไร้ใบตราสาร")
# context = preprocess_text(open("inputs/context/1345136.txt").read())
# result = question_answerer(question, context)

# print(result)

import pandas as pd

query = "13_1345136.txt"

df = pd.read_csv("inputs/sub_paragraph_58txt_wSameTopic_pos.csv")
print(df.subdoc_name == query)
print(type(df.loc[df.subdoc_name == query, "context"].iat[0]))
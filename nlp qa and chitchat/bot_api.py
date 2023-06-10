# ตัวอย่างของ Chatbot API
# run with
#   uvicorn --host 0.0.0.0 --reload --port 3000 bot_api:app

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pathlib import Path

# elastic search
from elastic_search.elastic_search import DocumentRetrieval

# classify

# qa
import pandas as pd
from qa.qa import preprocess_text  # SimpleTransformer
from transformers import pipeline
import torch

# chitchat
from chitchat.chitchat import ChitChat


CONTEXT_DIR = Path(r"inputs/context")

# document classifier


# elastic search
ES_HOSTS = "https://localhost:9200"
ES_USER = "elastic"
ES_PASS = "Pic1zs71iL1RWB3By=DC"
HTTP_CA_PATH = "elastic_search/http_ca.crt"
index_name = "document"

print("Loading Elastic Model")
document_retrieval = DocumentRetrieval(ES_HOSTS, ES_USER, ES_PASS, HTTP_CA_PATH)

# classify document
# add code


# question and answering
# QA_TYPE = "camembert"
# QA_PATH = "qa/weights/model_data_kaggle_iapp_squadv1_epochs40"
# QA_DEVICE = -1

print("Loading QA Model")
# question_answering = SimpleTransformer(QA_TYPE,
#                                        QA_PATH,
#                                        QA_DEVICE)
QA_THRESH = 1e-4
question_answerer = pipeline("question-answering", model="parinzee/mdeberta-cmsk-qa", device=torch.device("cuda:1"))
PREPROCESS_DF = pd.read_csv("inputs/sub_paragraph_58txt_wSameTopic_pos.csv")

# chit chat
print("Loading Chitchat Model")
chitchat= ChitChat()


print("Server Starting")
app = FastAPI()

@app.get("/chat")
async def echo(line: str):
    question = line
    
    # elastic search
    context_candidate = document_retrieval.search(question, index_name=index_name, thresh=50)
    print("context candidate", context_candidate)
    
    # classfy document
    # document_classifier = classifier_data(line)
    
    question_is_qa = len(context_candidate)
    if question_is_qa:  # qa
        sub_context_candidate = list(context_candidate.keys())[0]
        
        # context_path = CONTEXT_DIR / context_candidate
        # context = preprocess_text(open(context_path).read())
        
        context = PREPROCESS_DF.loc[PREPROCESS_DF.subdoc_name == sub_context_candidate, "context"].iat[0]
        context = preprocess_text(context)
        question = preprocess_text(question)
        
        result = question_answerer(question, context)
        print(f"Result:", result)
        result = result["answer"] if result["score"] >= QA_THRESH else ""
            
    else:  # chitchat
        result, conver_in, conver_out = chitchat.generate(question)
        
        print(f"Result:", result)
    return PlainTextResponse(result)


# def result(question):
#     context_candidate = document_retrieval.search(question)
    
#     is_qa = len(context_candidate)
    
#     if is_qa:  # qa
#         # qa process
#         pass
#     else:  # chit chat
#         # chit chat process
#         pass
    
    
    
#     return None

# def pipeline()
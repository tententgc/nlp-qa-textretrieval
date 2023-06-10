# ตัวอย่างของ Chatbot API
# run with
#   uvicorn --host 0.0.0.0 --reload --port 3000 bot_api:app
# 114.119.187.37
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from retrieval import retrieve_doc, doc_df
from question_answering import qa_predict

from classify_api import prediction
from chitchat import answering

import warnings
warnings.filterwarnings("ignore")

app = FastAPI()

@app.get("/chat")
async def echo(line: str):
    answer_mode = prediction(str(line))
    answer_mode # for classifly text results, qa or chitchat
    if answer_mode == "qa":
        doc_cids = retrieve_doc(line)
        answer = qa_predict(doc_cids, line)
    elif answer_mode == "chitchat":
        answer = answering(line)[0]
    return PlainTextResponse(answer)      # None = QA with no answer
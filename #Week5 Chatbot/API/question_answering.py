### import library
import pandas as pd 
import numpy as np
from pythainlp.util import normalize
from stop_words import get_stop_words
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus.common import thai_words
from pythainlp import Tokenizer,sent_tokenize,word_tokenize

import re
import string
import statistics
from typing import Counter
import glob
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
import logging
import torch
import os

from retrieval import doc_df

### setup model path
model_path = "./best_model_final"

### preprocess
def qa_preprocess(text):
    return text 

def model_qa(model_path):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    model_args_xlm = QuestionAnsweringArgs()

    model_args_xlm.learning_rate = 3e-5
    model_args_xlm.num_train_epochs = 10
    model_args_xlm.max_seq_length = 512
    model_args_xlm.doc_stride = 128
    model_args_xlm.overwrite_output_dir = True
    model_args_xlm.evaluate_during_training = True
    model_args_xlm.reprocess_input_data = False
    model_args_xlm.train_batch_size = 16
    model_args_xlm.n_best_size = 50
    model_args_xlm.max_answer_length = 100
    model_args_xlm.gradient_accumulation_steps = 1
    model_args_xlm.null_score_diff_threshold = 1

    model_xlm = QuestionAnsweringModel(
        "xlmroberta", model_path, args=model_args_xlm, use_cuda=torch.cuda.is_available()
    )
    return model_xlm


def qa_predict(doc_cids, line, model=model_qa(model_path)):
    context_to_pred = doc_df["context"].iloc[doc_cids[0]] 
    to_predict = [
        {
            "context": context_to_pred, ## context from retrieval
            "qas": [
                {
                    "question": line, ## question from main api
                    "id": 1,
                }
            ],
        }
    ]
    answers, probs = model.predict(to_predict)
    final_answer = answers[0]['answer'][0]
    return final_answer
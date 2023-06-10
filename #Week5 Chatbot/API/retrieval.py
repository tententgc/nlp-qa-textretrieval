# Document retrieval with BM25
import numpy as np
import pandas as pd
import os

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from rank_bm25 import BM25Okapi

th_stopwords = set(thai_stopwords())
context_folder = './context'

def preprocess(text, stopwords):
    tokens = word_tokenize(text, engine='newmm')
    tokens = [x for x in tokens if any(c.isalnum() for c in x)]  # eliminate spaces and symbols
    tokens = [x for x in tokens if x not in stopwords]
    tokens = [x for x in tokens if not x.isnumeric()]
    return tokens

doc_ls = []
for dirname, _, filenames in os.walk(context_folder):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            context = f.read()
        doc_ls.append([filename, context])

doc_df = pd.DataFrame(doc_ls, columns=['file', 'context'])
doc_df['cid'] = doc_df.index
tokenized_contexts = list(doc_df['context'].apply(lambda x: preprocess(x, th_stopwords)))
bm25 = BM25Okapi(tokenized_contexts, k1=0.9, b=0.2)
print('Document retrieval initialized!')
# print(doc_df)

def retrieve_doc(query, topk=5):
    tokenized_query = preprocess(query, th_stopwords)
    doc_scores = bm25.get_scores(tokenized_query)
    return list(np.argsort(doc_scores)[::-1][:topk])
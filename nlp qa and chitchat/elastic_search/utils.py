import pandas as pd
from tqdm import tqdm


def read_file(path):
    with open(path) as f:
        data = f.read().strip()
    return data


def create_elastic(es, df, index_name):
    mappings = {
        "properties": {
            "filename": {"type": "text", "analyzer": "trigrams"},
            "title": {"type": "text", "analyzer": "trigrams"},
            "context": {"type": "text", "analyzer": "trigrams"}
        }
    }

    settings = {
        "analysis": {
            "analyzer": {
                "trigrams": {
                    "tokenizer": "trigram_tokenizer",
                    "filter": [
                        "lowercase"
                    ]
                }
            },
            "tokenizer": {
                "trigram_tokenizer": {
                    "type": "ngram",
                    "min_gram": 3,
                    "max_gram": 3,
                    "token_chars": []
                }
            }
        }
    }

    es.indices.create(index=index_name, mappings=mappings, settings=settings)
    for i, row in tqdm(df.iterrows(), desc="Create Elastic Data"):
        doc = {
            "filename": row.filename,
            "title": row.title,
            "context": row.context
        }

        es.index(index=index_name, id=i, document=doc)


def elastic_seach(es, query, index_name):
    resp = es.search(
        index=index_name,
        body={"query": {
            "match": {
                "context": query
            }
        }
        }
    )
    return resp

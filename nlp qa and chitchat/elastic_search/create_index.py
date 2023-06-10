from pathlib import Path
from elasticsearch import Elasticsearch
from utils import read_file  # , create_elastic, elastic_seach
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

ES_HOSTS = "https://localhost:9200"
ES_USER = "elastic"
ES_PASS = "Pic1zs71iL1RWB3By=DC"
HTTP_CA_PATH = "http_ca.crt"
index_name = "document"

es = Elasticsearch(hosts=ES_HOSTS,
                   basic_auth=(ES_USER, ES_PASS),
                   ca_certs=HTTP_CA_PATH)


def elastic_seach(query, index_name):
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


def search_context(df, index_name, transform=None):
    df = df.copy()
    # filename_actual = []
    filename_search = []

    for i, row in tqdm(list(df.iterrows())):
        question = row.question

        if transform:
            question = transform(question)

        res = elastic_seach(question, index_name)

        filename_pred = None
        have_seach = len(res["hits"]["hits"])
        if have_seach:
            max_score = res["hits"]["max_score"]

            for hit in res["hits"]["hits"]:
                score = hit["_score"]
                if score == max_score:
                    search_file = hit["_source"]["filename"]

            filename_pred = search_file
        filename_search.append(filename_pred)
    df["search"] = filename_search
    return df


def create_elastic(df, index_name):

    mappings = {
        "properties": {
            "sub_filename": {"type": "text", "analyzer": "trigrams"},
            "filename": {"type": "text", "analyzer": "trigrams"},
            # "title": {"type": "text", "analyzer": "trigrams"},
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
                    "min_gram": 2,
                    "max_gram": 3,
                    "token_chars": []
                }
            }
        }
    }

    es.indices.create(index=index_name, mappings=mappings, settings=settings)
    for i, row in tqdm(df.iterrows(), desc="Creating Elastic Data"):
        doc = {
            "sub_filename": row.subdoc_name,
            "filename": row.doc_name,
            "context": row.context
        }

        es.index(index=index_name, id=i, document=doc)


def check_viz(df, col="check", title=None):
    # Set the plot style and figure size
    sns.set(style="white", rc={"figure.figsize": (10, 6)})

    g = sns.countplot(
        x=col, data=df, order=df[col].value_counts().index.to_list())
    g.set_xticklabels(g.get_xticklabels(), rotation=0)

    # Add value labels to the bars
    for p in g.patches:
        g.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.xlabel("Check")
    plt.ylabel("Count")
    plt.title(title)
    plt.savefig("test1.jpg")
    plt.show()


def main():
    # context_dir = Path(r"../inputs/context")

    # info = []
    # for p in sorted(list(context_dir.iterdir()), key=lambda f: int(Path(f).stem)):
    #     filename = p.name
    #     context = read_file(p)
    #     # context = preprocess_text(context)
    #     title = context.split("\n")[0]
    #     info.append([filename, title, context])

    # df = pd.DataFrame(info, columns=["filename", "title", "context"])
    # print(df.head())

    df = pd.read_csv(
        "/home/user/workspace/inputs/sub_paragraph_58txt_wSameTopic_pos.csv")

    try:
        create_elastic(df, index_name)
    except:
        print(f"{index_name} is exist!!! but creating new one")
        es.indices.delete(index=index_name)
        create_elastic(df, index_name)

    es.indices.refresh(index=index_name)

    print(es.indices.get_alias(index="*"))

    train_df = pd.read_csv("../inputs/train.csv")
    train_df = search_context(train_df, index_name=index_name, transform=None)
    train_df["check"] = train_df["file"] == train_df["search"]
    train_df["check"] = train_df.check.apply(
        lambda c: "correct" if c else "incorrect"
    )

    train_df.loc[train_df.search.isna(), "check"] = "unknown"
    train_df.to_csv("test1.csv")
    check_viz(train_df, title="Histogram of Check without Parin Preprocess")
    res = elastic_seach(
        "ใครเป็นผู้ออกตราสารหนี้ภาคเอกชน ไร้ใบตราสาร", index_name)
    
    print(res)


if __name__ == "__main__":
    main()
    # df = pd.read_csv("/home/user/workspace/inputs/sub_paragraph_58txt_wSameTopic_pos.csv")
    # print(df)

###############################################################
# first install library via "pip install -r requirements.txt" #
###############################################################
import pythainlp
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import thai_words
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words
from nltk.corpus import wordnet
import pickle

import re
import string
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")


import nltk
th_stop = tuple(thai_stopwords())
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()


# declare function
def split_word(text, remove_stop_word=False):
  tokens = word_tokenize(text, engine='newmm') # newmm is ditionary cutting
  # Remove stop words ภาษาไทย และภาษาอังกฤษ
  if remove_stop_word:
    tokens = [i for i in tokens if not i in th_stop and not i in en_stop]
  # หารากศัพท์ภาษาไทย และภาษาอังกฤษ
  # English
  tokens = [p_stemmer.stem(i) for i in tokens]
  
  # Thai
  tokens_temp=[]
  for i in tokens:
      w_syn = wordnet.synsets(i)
      if (len(w_syn)>0) and (len(w_syn[0].lemma_names('tha'))>0):
          tokens_temp.append(w_syn[0].lemma_names('tha')[0])
      else:
          tokens_temp.append(i)
  
  tokens = tokens_temp

  # ลบตัวเลข
  tokens = [i for i in tokens if not i.isnumeric()]
  # ลบช่องว่าง
  tokens = [i for i in tokens if not ' ' in i]
  return tokens

def clean_msg(msg):
    # ลบ text ที่อยู่ในวงเล็บ <> ทั้งหมด
    msg = re.sub(r'<.*?>','', msg)
    # ลบ hashtag
    msg = re.sub(r'#','',msg)
    # ลบ เครื่องหมายคำพูด (punctuation)
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c),'',msg)
    # ลบ separator เช่น \n \t
    msg = ' '.join(msg.split())
    return msg

def prediction(message):
    # load model
    model = pickle.load(open('model.pk', 'rb'))
    # load tfidf
    ws_tfidf = pickle.load(open('tfidf.pk', 'rb'))
    message = clean_msg(message)
    message = split_word(message, remove_stop_word=False)
    message = ' '.join(message)
    message = [message]
    message = ws_tfidf.transform(message)
    prediction = model.predict(message)
    txt_labels = ['chitchat', 'qa']
    return txt_labels[prediction[0]]


def main():
    df = pd.read_csv("training_data\Classify_data.csv")
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    df["labels"] = labelencoder.fit_transform(df["mode"])
    df["labels"] = labelencoder.fit_transform(df["mode"])
    x = df['question']
    y = df['labels']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    
    ws_tfidf = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 2), sublinear_tf=True)
    ws_vec = ws_tfidf.fit_transform(X_train)
    ws_vec_test = ws_tfidf.transform(X_test)

    # modeling
    
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=2, random_state=42)),
        ('svc',LinearSVC(random_state=42)),
        ('knn',KNeighborsClassifier(n_neighbors=2)),
        ('lgr',LogisticRegression(random_state=42))
    ]

    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(random_state=42),
        n_jobs=-1,
        verbose=0,
        cv=10
    )

    clf.fit(ws_vec, y_train)
    # input_text = [input('Enter your text: ')]
    # vec_test = ws_tfidf.transform(input_text)
    # pred = clf.predict(vec_test)
    # print('results = ', df['mode'].unique()[pred])
    pickle.dump(clf, open('model.pk', 'wb'))
    pickle.dump(ws_tfidf, open("tfidf.pk", "wb"))

if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')



df=pd.read_csv('movies.csv')

req_col=["title","genres","keywords","overview","cast"]
df=df[req_col]

df = df.dropna().reset_index(drop=True)

df['combined']=df['genres']+' '+df['cast']+' '+df['keywords']+' '+df['overview']

data=df[['title','combined']]

stop_words= set(stopwords.words('english'))

def preprocessing(text):
  text=re.sub(r"[^a-zA-Z1-9]"," ",text)
  text=text.lower()
  tokens=word_tokenize(text)
  tokens = [token for token in tokens if token not in stop_words]
  #[] making a new list by iterating through text and the first token is the return word which doesn't belong to the stopword
  return " ".join(tokens)

data['combined']=data['combined'].apply(preprocessing)

tfidf=TfidfVectorizer(max_features=5000)
tfidf_mat=tfidf.fit_transform(data['combined'],data['combined'])

cos_sim=cosine_similarity(tfidf_mat)


joblib.dump(data, 'data.pkl')
joblib.dump(tfidf_mat, 'tfidf_mat.pkl')
joblib.dump(cos_sim, 'cos_sim.pkl')
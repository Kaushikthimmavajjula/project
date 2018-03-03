import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import pickle as pk

df = pd.read_csv('./sample.csv',header = None)

df.columns = ['name','text']

df = df.dropna()
#vectorize documents 
vectorizer = TfidfVectorizer(max_features = 1000) 
doc_vecs = vectorizer.fit_transform(df.text.dropna())
#dimenstionality reduction
svd = TruncatedSVD(n_components=20, n_iter=10, random_state=42)
dv = svd.fit_transform(doc_vecs)
#clustering
kmeans = KMeans(n_clusters=5, random_state=42,n_jobs=8).fit(dv)

df['class'] = kmeans.labels_
#save models learnt 
with open('model_tfidf.pkl','wb') as f:
    pk.dump(vectorizer,f)

with open('model_svd.pkl','wb') as f:
    pk.dump(svd,f)

with open('model_km.pkl','wb') as f:
    pk.dump(kmeans,f)









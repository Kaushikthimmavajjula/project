
import pickle as ps
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from flask import Flask, jsonify, request

app = Flask(__name__)
with open('model_svd.pkl','rb') as f:
    svd = ps.load(f, encoding ='latin1')

with open('model_km.pkl','rb') as f:
    kmeans = ps.load(f,encoding ='latin1')
    
with open('model_tfidf.pkl','rb') as f:
    vectorizer = ps.load(f,encoding ='latin1')
    

classify = lambda x: str(kmeans.predict(svd.transform(vectorizer.transform([x])))[0])

@app.route('/service/<text>')
def get_classes(text):
    lable = 'class ' + classify(text)# insert any document text here
    return lable 
if __name__ == '__main__':
    app.run()






from flask import Flask, request, jsonify
from flask_cors import CORS

import time
from solution import solution
import uuid
import whisper
from autocorrect import Speller
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymorphy2
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
import pickle

count_test = 0
morph = pymorphy2.MorphAnalyzer()
spell = Speller('ru')
large_model = whisper.load_model("small")

def lemmatize(text):
    words = text.split()
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)
    return ' '.join(res)

dir_path_model = r'/root/hackathon-service-template'
is_model = 0# Iterate directory
for path in os.listdir(dir_path_model):
    if path == 'mnb2':
        is_model = 1

if not is_model:
    print('hello 1')
    dir_path_NotFraud = r'/root/hackathon-service-template/text/NotFraud'
    notfraud_text = []
    # Iterate directory
    for path in os.listdir(dir_path_NotFraud):
        # check if current path is a file
    #     if os.path.isfile(os.path.join(dir_path_NotFraud, path)):
    #         notfraud_text.append(path)
        with open(f'/root/hackathon-service-template/text/NotFraud/{path}', 'r') as r:
                # Преобразование объектов Python в данные 
                # JSON формата, а так же запись в файл 'data.json'
            text = json.load(r)
        notfraud_text.append(text)

    print('hello 2')
    dir_path_Fraud = r'/root/hackathon-service-template/text/Fraud'
    fraud_text = []
    # Iterate directory
    for path in os.listdir(dir_path_Fraud):
        # check if current path is a file
    #     if os.path.isfile(os.path.join(dir_path_NotFraud, path)):
    #         notfraud_text.append(path)
        with open(f'/root/hackathon-service-template/text/Fraud/{path}', 'r') as r:
                # Преобразование объектов Python в данные 
                # JSON формата, а так же запись в файл 'data.json'
            text = json.load(r)
        fraud_text.append(text)

    print('hello 3')
    nft = pd.DataFrame(notfraud_text)
    nft.columns = ['text']
    nft.loc[:, 'is_fraud'] = 0

    ft = pd.DataFrame(fraud_text)
    ft.columns = ['text']
    ft.loc[:, 'is_fraud'] = 1

    sdf = pd.concat((ft, nft))
    sdf.reset_index(inplace=True, drop=True)
    sdf.loc[:, "norm"] = sdf.apply(lambda x: lemmatize(spell(x.text)), axis=1)

    # extra_df = pd.read_csv('additional_data_450.csv')
    # extra_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # extra_notfraud = np.random.RandomState(42).choice(extra_df[extra_df.is_fraud==0].index, 9, 0)
    # extra_fraud = np.random.RandomState(42).choice(extra_df[extra_df.is_fraud==1].index, 5, 0)
    # new_extra_df = pd.concat([extra_df.loc[extra_fraud, :], extra_df.loc[extra_notfraud, :]])
    # bdf = pd.concat((new_extra_df, sdf))
    # bdf.reset_index(inplace=True, drop=True)

    X_train = sdf.norm
    y_train = sdf.is_fraud
    print('hello 4')
    vectorizer = CountVectorizer(min_df=0.0005, max_df=0.1)
    vec_data_train = vectorizer.fit_transform(X_train).toarray()
    multinomial_nb = MultinomialNB()
    multinomial_nb.partial_fit(vec_data_train, y_train, classes=(0,1))
else:
    print('hello 1')
    dir_path_NotFraud = r'/root/hackathon-service-template/text/NotFraud'
    notfraud_text = []
    # Iterate directory
    for path in os.listdir(dir_path_NotFraud):
        # check if current path is a file
    #     if os.path.isfile(os.path.join(dir_path_NotFraud, path)):
    #         notfraud_text.append(path)
        with open(f'/root/hackathon-service-template/text/NotFraud/{path}', 'r') as r:
                # Преобразование объектов Python в данные 
                # JSON формата, а так же запись в файл 'data.json'
            text = json.load(r)
        notfraud_text.append(text)

    print('hello 2')
    dir_path_Fraud = r'/root/hackathon-service-template/text/Fraud'
    fraud_text = []
    # Iterate directory
    for path in os.listdir(dir_path_Fraud):
        # check if current path is a file
    #     if os.path.isfile(os.path.join(dir_path_NotFraud, path)):
    #         notfraud_text.append(path)
        with open(f'/root/hackathon-service-template/text/Fraud/{path}', 'r') as r:
                # Преобразование объектов Python в данные 
                # JSON формата, а так же запись в файл 'data.json'
            text = json.load(r)
        fraud_text.append(text)

    print('hello 3')
    nft = pd.DataFrame(notfraud_text)
    nft.columns = ['text']
    nft.loc[:, 'is_fraud'] = 0

    ft = pd.DataFrame(fraud_text)
    ft.columns = ['text']
    ft.loc[:, 'is_fraud'] = 1

    sdf = pd.concat((ft, nft))
    sdf.reset_index(inplace=True, drop=True)
    sdf.loc[:, "norm"] = sdf.apply(lambda x: lemmatize(spell(x.text)), axis=1)

    # extra_df = pd.read_csv('additional_data_450.csv')
    # extra_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # extra_notfraud = np.random.RandomState(42).choice(extra_df[extra_df.is_fraud==0].index, 9, 0)
    # extra_fraud = np.random.RandomState(42).choice(extra_df[extra_df.is_fraud==1].index, 5, 0)
    # new_extra_df = pd.concat([extra_df.loc[extra_fraud, :], extra_df.loc[extra_notfraud, :]])
    # bdf = pd.concat((new_extra_df, sdf))
    # bdf.reset_index(inplace=True, drop=True)

    X_train = sdf.norm
    y_train = sdf.is_fraud
    print('hello 4')
    vectorizer = CountVectorizer(min_df=0.0005, max_df=0.1)
    vec_data_train = vectorizer.fit_transform(X_train).toarray()
    with open('mnb2', 'rb') as handle:
        multinomial_nb = pickle.load(handle)
    print("Модель вгружена")
print("Запущено")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = set(["wav"])

vec_date = []

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/test-sample", methods=["POST"])
def test_sample():
    global count_test
    count_test += 1
    # check if the post request has the file part
    if "sample" not in request.files:
        response = jsonify({"message": "NO sample file in the request"})
        response.status = 400
        return response

    file = request.files["sample"]
    uid = uuid.uuid4()
    file.save(str(uid))
    file_path = str(uid)

    if not allowed_file(file.filename):
        response = jsonify({"message": "Only .wav file type is allowed"})
        response.status = 400
        return response
    res = solution(large_model, vectorizer, multinomial_nb, file_path, count_test, vec_date, is_model)
    vec_date.append([count_test, res[0], res[1]]) 
    response = jsonify({"result": res[1]})
    response.status = 200
    if not is_model:
        if count_test == 20:
            with open('mnb2', 'wb') as to_write:
                pickle.dump(multinomial_nb, to_write)
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

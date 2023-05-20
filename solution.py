from werkzeug.datastructures import FileStorage
import numpy as np
import pymorphy2
from autocorrect import Speller
import pandas as pd
import os
import psycopg2
import time

def solution(large_model, vectorizer, multinomial_nb, wav_file, count, vec_date, is_model) -> int:
    print("in solution")
    if not is_model:
        time.sleep(15)
        print("got sleep")
        conn = psycopg2.connect(host='localhost',
        database='postgres',
        user='postgres', password='12345678', sslmode='require')
        cur = conn.cursor()
        cur.execute('select * from checker;')
        conn.commit()
        data = np.array(cur.fetchall())
        if len(data):
            print("psql data not null")
            num_test = data[-1, :][1]
            ans = data[-1, :][2]
            if len(vec_date):
                print("vec not null")
                vec_d = vec_date[-1][1]
                if not isinstance(vec_d, int):
                    num_test_d = vec_date[-1][0]
                    res_d = vec_date[-1][2]
                    print(num_test_d, num_test)
                    if num_test_d == num_test:
                        print(f"Я вас услышала, номера тестов совпали, номер теста:{num_test}\n")
                        multinomial_nb.partial_fit(vec_d, np.array(res_d).reshape(1, ) if ans else np.array(not res_d).reshape(1, ))
        conn.close()
    try:
        morph = pymorphy2.MorphAnalyzer()
        spell = Speller('ru')
        def lemmatize(text):
            words = text.split()
            res = list()
            for word in words:
                p = morph.parse(word)[0]
                res.append(p.normal_form)
            return ' '.join(res)
        res_large = large_model.transcribe(wav_file)
        text = res_large['text']
        text_df = pd.DataFrame([text])
        text_df.columns = ['text']
        text_df.loc[:, "norm"] = text_df.apply(lambda x: lemmatize(spell(x.text)), axis=1)
        X_test = text_df.norm
        vec_data_test = vectorizer.transform(X_test).toarray()
        res = [vec_data_test, int(multinomial_nb.predict(vec_data_test)[0])]
    except:
        res = [0, int(np.random.uniform(low=0.0, high=1.0))]
    os.remove(wav_file)
    return res


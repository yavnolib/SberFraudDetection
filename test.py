import requests
import json
import os
import time

fraud_path = "./train-samples/Fraud"
not_fraud_path = "./train-samples/NotFraud"

fraud_files = [os.path.join(fraud_path,f) for f in os.listdir(fraud_path)]
not_fraud_files = [os.path.join(not_fraud_path,f) for f in os.listdir(not_fraud_path)]

print("Testing fraud files from: {}".format(fraud_path))
start_time = time.time()

tn = 0
# test fraud audio files
for file in fraud_files:
    files = {'sample': open(file,'rb')}
    values = {}
    response = requests.post("http://0.0.0.0:8080/test-sample",files=files, data=values)
    tn += 1 if json.loads(response.text)["result"]==1 else 0

tp = 0

# test non fraud audio files
for file in not_fraud_files:
    files = {'sample': open(file,'rb')}
    values = {}
    response = requests.post("http://0.0.0.0:8080/test-sample",files=files, data=values)
    tp += 1 if json.loads(response.text)["result"]==0 else 0

print("Test done!")
print("Fraud: {}/{} NonFraud: {}/{}".format(tn,len(fraud_files),tp,len(not_fraud_files)))
print("Elapsed time:{} sec".format(time.time()-start_time))

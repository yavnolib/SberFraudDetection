# SberFraudDetection
An algorithm for recognizing fraudulent calls, written as part of a joint hackathon between Sberbank and MIPT.

# Algorithm
* Getting a speech file via json and apply the Whisper "small" speech-recognition model to this file.
* After that, we auto-correct the received by Whisper text data using the autocorrection library.
* Normalise our text
* Apply CountVectorizer
* Partial MultinomialNB on train-samples data
* Run server and get audio. We use the Postgres database to compose feedback and check our predicts. If predicted value does not match with target value we use partial fit again.

During the hackathon received:
* accuracy on public tests (60): 85%
* accuracy on private tests (30): 73%

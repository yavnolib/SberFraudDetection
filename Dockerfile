FROM python:3.9 

ENV PYTHONUNBUFFERED=1

ADD app.py .
ADD solution.py .
ADD requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "./app.py"] 
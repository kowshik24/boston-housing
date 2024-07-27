FROM python:3.7-slim

COPY ./requirements.txt /webapp/requirements.txt

WORKDIR /webapp

RUN pip install -r requirements.txt

COPY webapp /webapp

CMD ["python", "app.py"]
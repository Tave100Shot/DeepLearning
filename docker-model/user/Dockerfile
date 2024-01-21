FROM amazon/aws-lambda-python:3.8

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip
RUN /var/lang/bin/python3.8 -m pip install joblib

WORKDIR /var/task/docker-model

COPY inference.py /var/task/
COPY . .

RUN pip install -r requirements.txt

CMD ["inference.handler"]
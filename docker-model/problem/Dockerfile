FROM amazon/aws-lambda-python:3.8


RUN /var/lang/bin/python3.8 -m pip install --upgrade pip


WORKDIR /var/task/thefolder

COPY test.py /var/task/
COPY . .

RUN pip install -r requirements.txt

CMD ["test.handler"]
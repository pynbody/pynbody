FROM python:3.11

WORKDIR /code

COPY . .

RUN pip install .[tests]

CMD [ "bash" ]


FROM python:3.10-slim
WORKDIR /app
COPY ./requirements.txt /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . /app
EXPOSE 5000
ENTRYPOINT FLASK_APP=driver_code.py flask run --host=0.0.0.0
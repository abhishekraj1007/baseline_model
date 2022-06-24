FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENTRYPOINT FLASK_APP=driver_code.py flask run --host=0.0.0.0

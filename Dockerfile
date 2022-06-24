FROM python:3.10-slim
WORKDIR .
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["python3", "driver_code.py"]
ENTRYPOINT FLASK_APP=driver_code.py flask run --host=0.0.0.0

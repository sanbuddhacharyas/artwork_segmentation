From python:3.8-slim

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx 

RUN pip install --upgrage pip
RUN install --no-cache-dir -r requirements.tx

EXPOSE 5000

CMD ["python", "./application.py"]
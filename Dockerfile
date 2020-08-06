FROM python:3.6
WORKDIR /get_emb
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get -y install cmake
RUN pip install -r ./requirements.txt
EXPOSE 80
COPY . .
CMD "python" "app.py"


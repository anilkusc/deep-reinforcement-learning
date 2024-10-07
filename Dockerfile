FROM python:3.12.3

WORKDIR /app
RUN apt update && apt install swig -y
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN chmod -R 777 /app
#ENTRYPOINT sleep infinity
ENTRYPOINT [ "python","-u","./main.py" ]
# sudo docker run -dit --name anil-test -v /app/models:/volumes/models --runtime nvidia --gpus all anilkuscu95/rl-racing-test
#sudo docker run -dit --runtime nvidia --gpus all test
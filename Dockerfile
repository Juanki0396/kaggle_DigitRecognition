FROM python:3.8.10-slim

#Creating a dir where hosting the code
WORKDIR /myapp

#Setting up the python enviroment
COPY requirements requirements

RUN ["mkdir" , "./data"]
RUN ["pip", "install", "--upgrade", "pip"]
RUN ["pip", "install", "-r", "requirements"]

#Copy the necessary code for inference
COPY src src
COPY models/Adam_ep_5_lr_0.001_batch_64/model models/Adam_ep_5_lr_0.001_batch_64/model
COPY inference.py inference.py

#Defining the entrypoint
ENTRYPOINT ["python", "inference.py", "--csvPath"]

CMD ["data/test.csv"]
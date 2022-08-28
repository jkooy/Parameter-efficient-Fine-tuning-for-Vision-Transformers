FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
COPY requirements.txt /tmp
WORKDIR /tmp
RUN apt-get update && apt-get install -y git
RUN pip install -r requirements.txt
RUN pip install psutil
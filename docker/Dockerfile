FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt install python3-pip -y
RUN apt-get install -y git

RUN pip3 install networkx==2.8.8
RUN pip3 install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install ultralytics==8.4.6
RUN pip3 install protobuf==3.20.1

RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y

RUN python3 -m pip install supervisely==6.73.486

RUN pip3 install requests==2.28.2 urllib3==1.26.15
RUN pip3 install packaging==21.3

RUN apt-get update && apt-get -y install curl

RUN apt -y install wireguard iproute2
RUN apt-get -y install wget
RUN apt-get install nano

RUN pip3 install ruamel.yaml==0.17.21
RUN pip3 install git+https://github.com/openai/CLIP.git

LABEL python_sdk_version="6.73.486"
FROM pytorch/pytorch:latest

RUN git clone https://github.com/WiseDoge/CoupletAI.git
RUN pip install flask
RUN apt-get update 
RUN apt-get install wget
RUN cd CoupletAI && wget https://github.com/wb14123/couplet-dataset/releases/download/1.0/couplet.tar.gz && tar -zxvf couplet.tar.gz && rm -rf couplet.tar.gz 
WORKDIR /workspace/CoupletAI

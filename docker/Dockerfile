FROM pytorch/pytorch:latest

RUN git clone https://github.com/WiseDoge/CoupletAI.git
RUN pip install flask
RUN apt-get update 
RUN apt-get install wget
RUN cd CoupletAI && wget https://github.com/wb14123/couplet-dataset/releases/download/1.0/couplet.tar.gz && tar -zxvf couplet.tar.gz && rm -rf couplet.tar.gz 
#RUN cd CoupletAI && mkdir output && cd output && wget https://github.com/WiseDoge/CoupletAI/releases/download/v1.0/cnn_lstm_att_20.pkl
WORKDIR /workspace/CoupletAI

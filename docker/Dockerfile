FROM pytorch/pytorch:latest

RUN git clone https://github.com/WiseDoge/CoupletAI.git
RUN pip install flask
RUN cd CoupletAI && mkdir output && cd output && wget https://github.com/WiseDoge/CoupletAI/releases/download/v1.0/cnn_lstm_att_20.pkl
RUN python webdemo.py
WORKDIR /workspace
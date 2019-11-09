import os

import torch
from flask import Flask, request, render_template

import config
from qa import create_qa_context

app = Flask(__name__)

device = torch.device('cpu')
output_dir = config.ouput_dir
vocab_path = f'./{config.data_dir}/vocabs'
model_path = max(os.listdir(output_dir))
ctx = create_qa_context(f'./{output_dir}/{model_path}', vocab_path, config.embed_dim, config.hidden_dim, device)


@app.route('/<coupletup>')
def api(coupletup):
    return ctx.predict(coupletup)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    coupletup = request.form.get("coupletup")
    coupletdown = ctx.predict(coupletup)
    return render_template("index.html", coupletdown=coupletdown)


if __name__ == '__main__':
    app.run(host='0.0.0.0')

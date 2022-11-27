import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# initialize tokenizer and model from pretrained GPT2 model


from flask import jsonify, make_response
from flask import Flask, redirect, url_for, request

app = Flask(__name__)


@app.route('/<name>', methods=['GET'])
def getData(name):
	if request.method == 'GET':
            sequence =  name
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')


            inputs = tokenizer.encode(sequence, return_tensors='pt')
            outputs = model.generate(inputs, max_length=200, do_sample=True)

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("aa"+text)
            d = {"data": text}
            return make_response(jsonify(d), 200)
     


if __name__ == '__main__':
	app.run(debug=True)

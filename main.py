from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def hello_world():
    return render_template('house.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        int_features = [pd.to_numeric(x) for x in request.form.values()]

        print(int_features)

        fea = np.reshape(int_features, (1, -1))
        pred = model.predict(fea)
        print(pred)
        ans = round((pred[0]), 2)
        return render_template('success.html', pred=ans)
    else:
        return render_template('error.html')


@app.route("/success")
def success(ans=None):
    if ans != None:
        return render_template('success.html', pred=ans)
    else:
        return render_template('error.html')


@app.route("/error")
def error():
    return render_template('error.html')


app.run(debug=True)

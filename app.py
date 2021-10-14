from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/prediction_completed', methods=['POST']) 
def home():
    sepal_length = request.form['sepal_length']
    sepal_width = request.form['sepal_width']
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']
    pred_array = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    pred = model.predict(pred_array)
    pred_proba = model.predict_proba(pred_array)
    return render_template('prediction.html', data=pred, probabilities = pred_proba)


if __name__ == "__main__":
    app.run(debug=True)

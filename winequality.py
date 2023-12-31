import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('Wine_RFreg_Model.pkl', 'rb'))
scaler = pickle.load(open('wine_scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    reshaped_data = np.array(data).reshape(1, -1)
    final_input = scaler.transform(reshaped_data)
    print(f'Final input: {final_input}')
    output = regmodel.predict(final_input)[0]
    return render_template('results.html', prediction = str(output))


if __name__ == '__main__':
    app.run(debug=True)
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app=Flask(__name__)

model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "age","sex","cp","trestbps", "chol", "fbs",
                       "restecg", "thalach", "exang", "oldpeak", "slope","ca",
                        "thal"]

   
    prediction = model.predict(features_value)
    output= round(prediction[0],2)

    if output == 0:
        res_val = "** heart disease **"
    else:
        res_val = "no heart disease "
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))
if __name__ == "__main__":
    app.run(debug=True)
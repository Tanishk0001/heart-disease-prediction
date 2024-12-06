from flask import Flask, render_template, request
import numpy as np
import joblib
from heart_prediction_model import predict_heart_disease  

app = Flask(__name__)

print(app.static_folder)  

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        name = request.form['name']
        age = int(request.form['age'])
        chest_pain = int(request.form['chest-pain'])  
        thalach = int(request.form['thalach'])

        input_data = np.array([[age, chest_pain, thalach]])

        prediction = predict_heart_disease(input_data)

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
        return render_template('index.html', prediction_result=result, name=name)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
import numpy as np
import json
import tensorflow as tf

app = Flask(__name__)

# Load the Keras model
model = tf.keras.models.load_model('data.h5')

# Define a function to preprocess the input data for the model
def preprocess_input(data):
    # Convert the input data to a NumPy array
    input_data = np.array([[
        data['total_rooms'],
        data['distance'],
        data['office_space'],
        data['university_students'],
        data['income'],
        data['distance_center']
    ]])

    # Normalize the input data
    input_data = (input_data - np.mean(input_data, axis=0)) / np.std(input_data, axis=0)

    return input_data

@app.route('/')
def index():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        input_data = {
            'total_rooms': float(request.form['total_rooms']),
            'distance': float(request.form['distance']),
            'office_space': float(request.form['office_space']),
            'university_students': float(request.form['university_students']),
            'income': float(request.form['income']),
            'distance_center': float(request.form['distance_center'])
        }

        # Preprocess the input data for the model
        input_data = preprocess_input(input_data)

        # Use the model to make a prediction
        prediction = model.predict(input_data)[0][0]

        htmlstr = f"<html><body>經過數據分析模型比對，營業利潤的預測結果為 {prediction:.2f} </body></html>"
        return htmlstr

    return render_template("form.html")

if __name__ == "__main__":
    app.run()

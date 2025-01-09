from flask import Flask, render_template, request,jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load dataset and model
data = pd.read_csv("cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    # Get unique locations from the dataset
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = (request.form.get('total_sqft'))
    
    # Create input DataFrame
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Make a prediction
    prediction = pipe.predict(input_data)[0] * 1e5
    
    # Return the prediction as a string
    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
## Model Deployment with FastAPI
# Note: This code is intended to be run in a Jupyter notebook environment.

# Environment Setup
pip install numpy pandas scikit-learn fastapi uvicorn joblib
pip install requests

# ----------------------------------------------------------------------------------------------- #

## PREPARE AND TRAIN THE MODEL

%%writefile ml.py
# This code will be written to the 'ml.py' file (Jupyter magic function)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler (serialized model)
dump(model, 'breast_cancer_model.joblib')
dump(scaler, 'scaler.joblib')
print("Successfuly saved the model and scaler joblib files.")

# ----------------------------------------------------------------------------------------------- #

# Run the following cell, you should see `Successfuly saved the model and scaler joblib files.`
%run ml.py

# ----------------------------------------------------------------------------------------------- #

## SETTING UP FAST API
## Set up endpoint to receive new data and make predictions

%%writefile main.py
from fastapi import FastAPI, HTTPException  # Import necessary components from FastAPI and HTTPException for error handling.
from pydantic import BaseModel  # Import BaseModel from Pydantic to define data models.
from typing import List  # Import List from typing to specify data types in data models.
from joblib import load  # Import load function from joblib to load pre-trained models.
import datetime

app = FastAPI()  # Create an instance of FastAPI to define and manage your web application.

# Load pre-trained model and scaler objects from disk. These are used for making predictions and scaling input data, respectively.
model = load('breast_cancer_model.joblib')
scaler = load('scaler.joblib')

# Define a data model for incoming prediction requests using Pydantic.
# This model ensures that data received via the API matches the expected format.
class QueryData(BaseModel):
    features: List[float]  # Define a list of floating point numbers to represent input features for prediction.

# Defining the root endpoint (IMPORTANT - serves as a welcome-orientation point for accessing API)
@app.get("/")
async def read_root():
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time according to your preference
    formatted_datetime = now.strftime("%d-%m-%Y %H:%M:%S")  # Example format (YYYY-MM-DD HH:MM:SS)

    # Print the formatted date and time
    return f"Hello, it is now {formatted_datetime}"

# Decorator to create a new route that accepts POST requests at the "/predict/" URL.
# This endpoint will be used to receive input data and return model predictions.
# Declaring async before a function definition is a way to handle asynchronous operations in FastAPI. 
# It allows the server to handle many requests efficiently by not blocking the server during operations 
# like network calls or while waiting for file I/O.

@app.post("/predict/")
async def make_prediction(query: QueryData):
    try:
        # The input data is received as a list of floats, which needs to be scaled (normalized) using the previously loaded scaler.
        scaled_features = scaler.transform([query.features])
        
        # Use the scaled features to make a prediction using the loaded model.
        # The model returns a list of predictions, and we take the first item since we expect only one.
        prediction = model.predict(scaled_features)
        
        # Return the prediction as a JSON object. This makes it easy to handle the response programmatically on the client side.
        return {"prediction": int(prediction[0])}
    except Exception as e:
        # If an error occurs during the prediction process, raise an HTTPException which will be sent back to the client.
        raise HTTPException(status_code=400, detail=str(e))

# ----------------------------------------------------------------------------------------------- #

# RUNNING THE FastAPI API SERVER (Launch and run in terminal)
uvicorn main:app --reload 		# Note: For production, omit --reload command

# Access local server:
# http://127.0.0.1:8000

# ----------------------------------------------------------------------------------------------- #

## TESTING THE API
## Script to send POST requests with dummy data to your API
%%writefile test_api.py
import requests
import json

# Define the API endpoint
url = "http://127.0.0.1:8000/predict/"

# Sample data with 30 dummy feature values
data = {
    "features": [
        1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01, 3.001e-01, 
        1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01, 8.589e+00, 1.534e+02, 
        6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02, 3.003e-02, 6.193e-03, 2.538e+01, 
        1.733e+01, 1.846e+02, 2.019e+03, 1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 
        4.601e-01, 1.189e-01
    ]
}

# Send a POST request
response = requests.post(url, json=data)

# Print the response
print("Status Code:", response.status_code)
print("Response Body:", response.json())

# ----------------------------------------------------------------------------------------------- #

%run test_api.py

# ----------------------------------------------------------------------------------------------- #

%%writefile dummy.csv
1.799e+01,1.038e+01,1.228e+02,1.001e+03,1.184e-01,2.776e-01,3.001e-01,1.471e-01,2.419e-01,7.871e-02,1.095e+00,9.053e-01,8.589e+00,1.534e+02,6.399e-03,4.904e-02,5.373e-02,1.587e-02,3.003e-02,6.193e-03,2.538e+01,1.733e+01,1.846e+02,2.019e+03,1.622e-01,6.656e-01,7.119e-01,2.654e-01,4.601e-01,1.189e-01

# ----------------------------------------------------------------------------------------------- #

%%writefile test_api_csv.py
import requests
import json
import pandas as pd

# Define the API endpoint
url = "http://127.0.0.1:8000/predict/"

# Read data from dummy.csv using pandas
try:
  data = pd.read_csv("dummy.csv", header=None).values.tolist()
except FileNotFoundError:
  print("Error: File 'dummy.csv' not found. Please ensure the file exists.")
  exit(1)

# Ensure data contains 30 features
if len(data[0]) != 30:
  print("Error: 'dummy.csv' does not contain 30 features. Please check the file format.")
  exit(1)

# Convert data to a list of floats (assuming each row has 30 features)
data = [float(value) for value in data[0]]  # Access the first row

# Prepare data dictionary
data = {
  "features": data
}

# Send a POST request
response = requests.post(url, json=data)

# Print the response
print("Status Code:", response.status_code)
print("Response Body:", response.json())

# ----------------------------------------------------------------------------------------------- #

%run test_api_csv.py

# ----------------------------------------------------------------------------------------------- #

# Expected Output:
# Status Code:
# 	200 indicates that the HTTP request was successfully received, understood, and processed by the server.
# Response Body:
# 	This part will show the prediction result returned by the machine learning model.
# 	In this case, 0 indicates "no cancer detected" (depending on how the labels are encoded).

# ----------------------------------------------------------------------------------------------- #

# Remember to stop the Uvicorn server: Ctrl + C

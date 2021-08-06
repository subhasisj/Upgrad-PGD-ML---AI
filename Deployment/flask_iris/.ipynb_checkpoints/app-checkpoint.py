# import Flask class from the flask module
from flask import Flask, request
from sklearn.externals import joblib
import numpy as np

# Create Flask object to run
app = Flask(__name__)

# Load the model from the file
iris_model = joblib.load('model/iris_model.pkl')

@app.route('/')
def home():
    return "Iris Model Deployment!!"

@app.route('/predict')
def predict():
    # Get values from browser
    sepal_length = request.args['sepal_length']
    sepal_width = request.args['sepal_width']
    petal_length = request.args['petal_length']
    petal_width = request.args['petal_width']
    
    print(sepal_length)

    test_inp = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, 4)
    class_predicted = int(iris_model.predict(test_inp)[0])
    output = "Predicted Iris Class: " + str(class_predicted)

    return (output)


if __name__ == "__main__":
    # Start Application
    app.run()
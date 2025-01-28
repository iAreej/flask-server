import pandas as pd # type: ignore
from flask import Flask, jsonify,request # type: ignore
from flask_cors import CORS # type: ignore
import pickle
from sklearn.preprocessing import StandardScaler # type: ignore


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Load the dataset that was used to train the model
train_data = pd.read_csv('Smoker_data.csv')

# Create a StandardScaler instance and fit it to the training data
scaler = StandardScaler()
scaler.fit(train_data[['nicotine', 'tar', 'carbonmonoxide', 'formaldehyde', 'benzene', 'acetone', 'toluene']])


model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
   return 'Hello, World!'


@app.route("/api/users", methods=['GET'])
def users():
    print('users data')
    return jsonify(
        {
            "users": [
                'arpan',
                'zach',
                'jason',
                ]
        }
    )


@app.route('/api/predict', methods=['POST'])
def predict():
    print('hehe')
    data = request.get_json()  # Get the JSON data sent in the request
    print(data)  # Print the received data (for debugging)
    try:
      #Check if the data is in the correct format
     if not isinstance(data, list) or len(data) != 7:
          raise ValueError("Data must be a list of 7 thresholds.")

        # Convert the received data into a DataFrame
     input_df = pd.DataFrame([data], columns=['nicotine', 'tar', 'carbonmonoxide', 'formaldehyde', 'benzene', 'acetone', 'toluene'])
    
        # Standardize the input DataFrame
     input_standardized = scaler.transform(input_df)
        # Make a prediction using the standardized input
     prediction = model.predict(input_standardized)
     predicted_class = "Smoked" if prediction[0] == 1 else "Not Smoked"
     print(predicted_class)
     return jsonify({'message': 'Thresholds received successfully!', 'result': predicted_class})
    
    except Exception as e:
      print(f"Error: {e}")
      return jsonify({'message': 'An error occurred', 'error': str(e)}), 500
    




   

if __name__ == "__main__":
   app.run(debug=True ,host="0.0.0.0", port=5000)

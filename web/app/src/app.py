from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

path = "./my_acrobot_models/acrobot_model_policy_v70.h5"

model = tf.keras.models.load_model(path)

# Routes for your HTML pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/doublependulum')
def double_pendulum():
    return render_template('doublependulum.html')

@app.route('/pendtest')
def pend_test():
    return render_template('pendtest.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data from the client
        data = request.get_json()
        state = data['state']  # Assuming the state is sent as a list
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        
        # Predict using the model
        #prediction = model.predict(state_tensor).tolist()
        prediction = "hi"
        
        # Return the prediction as JSON
        return jsonify({'action': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
        # Test prediction before starting the server
    print("Testing prediction functionality...")
    
    # Create a sample state (adjust dimensions according to your model's input shape)
    test_state = [0.0, 0.0, 0.0, 0.0]  # Example state vector
    test_data = {'state': test_state}
    # Load the TensorFlow model

    import os


   # path = "./my_acrobot_models/acrobot_model_policy_v70.h5"
    print("File exists:", os.path.exists(path))

    try:
        # Convert to tensor
        test_tensor = tf.convert_to_tensor([test_state], dtype=tf.float32)
        
        # Make prediction
        test_prediction = model.predict(test_tensor)
        print("Test prediction successful!")
        print("Input state:", test_state)
        print("Model prediction:", test_prediction.tolist())
        
    except Exception as e:
        print("Error during test prediction:", str(e))
    
    # Start the Flask server
    print("\nStarting Flask server...")
    app.run(host='0.0.0.0', port=8080)

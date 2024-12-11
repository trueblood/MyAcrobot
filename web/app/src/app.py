from flask import Flask, render_template, request, jsonify
import torch
import psutil
import os
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x): # the brain of the agent, pushes the data through the 3 layers, pytorch auto does this
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

app = Flask(__name__)

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
        # # Receive JSON data from the client
        # data = request.get_json()
        # state = data.get('state')
        
        # if state is None:
        #     return jsonify({'error': 'No state provided in request'}), 400

        # state_dim = len(state)  # Number of features in the state (6 in this case)
        # action_dim = 3  # Number of possible actions (adjust as per your environment)

        # # Load the model
        # model = DQN(state_dim, action_dim)
        # model.load_state_dict(torch.load('acrobot_model_policy_v500.pth'))
        # model.eval()

        # # Convert the example state to a tensor
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension

        # # Perform inference to get Q-values
        # with torch.no_grad():  # No gradients needed for inference
        #     q_values = model(state_tensor)  # Forward pass to get Q-values
        #     best_action = torch.argmax(q_values).item()  # Get the best action (highest Q-value)
    
        # # Output the Q-values and the best action

        # print(f"Q-values: {q_values.numpy()}")
        # print(f"Best action: {best_action}")

        # Return the results as JSON
        return jsonify({
            'state': 'Reteturn Tim'
            #'state': state
            # 'q_values': q_values.tolist(),
            # 'action': best_action
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import requests
    import json

    # Start the Flask server
    print("\nStarting Flask server...")
    
    # Run the Flask app in a separate thread if needed
    from threading import Thread
    server_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=8080))
    server_thread.start()

    # Give the server a moment to start
    import time
    time.sleep(2)

    # Define the example JSON body
    example_json = {
        "state": [0.06528811, 0.99786645, -0.03661686, -0.9993294, 0.02972009, 1.3947774]
    }

    # Send a POST request to the server
    url = "http://localhost:8080/predict"
    response = requests.post(url, data=json.dumps(example_json), headers={"Content-Type": "application/json"})

    # Print the server's response
    if response.status_code == 200:
        print("Response from server:")
        print(response.json())
    else:
        print("Error:", response.status_code, response.text)


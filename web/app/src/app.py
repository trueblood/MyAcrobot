# from flask import Flask, render_template

# app = Flask(__name__)

# # Routes for your HTML pages
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

# @app.route('/doublependulum')
# def double_pendulum():
#     return render_template('doublependulum.html')

# @app.route('/pendtest')
# def pend_test():
#     return render_template('pendtest.html')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
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
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('message_from_server', {'data': 'Welcome to the WebSocket server!'})

@socketio.on('message_from_client')
def handle_message_from_client(message):
    try:
        print(f"Message from client: {message}")
        
        # Extract state from the message
        state = message.get('state')
        
        if state is None:
            emit('message_from_server', {'error': 'No state provided in request'})
            return

        # Convert state to proper format if it's not already a list
        if isinstance(state, str):
            try:
                state = eval(state)  # Be careful with eval - ensure input is sanitized
            except:
                emit('message_from_server', {'error': 'Invalid state format'})
                return

        state_dim = len(state)
        action_dim = 3

        # Load the model
        model = DQN(state_dim, action_dim)
        model.load_state_dict(torch.load('static/model/acrobot_model_policy_v4000.pth'))
        model.eval()

        # Convert the state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            q_values = model(state_tensor)
            best_action = torch.argmax(q_values).item()

        # Convert q_values to regular Python list for JSON serialization
        q_values_list = q_values.numpy().tolist()[0]

        # Emit results back to client
        response = {
            'q_values': q_values_list,
            'best_action': best_action,
            'original_state': state
        }
        
        emit('message_from_server', {'data': response}, broadcast=True)
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        emit('message_from_server', {'error': f'Server error: {str(e)}'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    
# Routes for your HTML pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/testsocket')
def index():
    return render_template('testsocket.html')

if __name__ == '__main__':
    print("\nServer running!")
    print("WebSocket URL: http://localhost:8077/testsocket")
    print("Click the URL above to open in your browser\n")
    socketio.run(app, host='0.0.0.0', port=8077)

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
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import torch
import torch.nn as nn
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path
import os

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
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///myacrobot.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////app/db/myacrobot.db'


app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
#socketio = SocketIO(app)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable CORS for SocketIO

# Ensure /app/db directory exists
db_dir = '/app/db'
Path(db_dir).mkdir(parents=True, exist_ok=True)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)

class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    score = db.Column(db.Integer, nullable=False)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('message_from_server', {'data': 'Welcome to the WebSocket server!'})

@socketio.on('message_from_client')
def handle_message_from_client(message):
    try:
      #  print(f"Message from client: {message}")
        
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

@app.route('/api/save-messages', methods=['POST'])
def save_messages():
    try:
        data = request.get_json()
        messages = data.get('messages')
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        message = Message(content=messages)
        db.session.add(message)
        db.session.commit()
        
        # Here you can add code to save the messages to a database or file
        # For example, saving to a SQLite database

        # Example: Save messages to a file (you can replace this with database logic)
     #   with open('saved_messages.txt', 'w') as f:
     #       f.write(messages)
        
        return jsonify({'success': True, 'message': 'Messages saved successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/save-score', methods=['POST'])
def save_score():
    try:
        data = request.get_json()
        name = data.get('name')
        curScore = data.get('curScore')
        # if not name or not curScore:
        #     return jsonify({'error': 'Name and score are required'}), 400
        # with open('saved_scores.txt', 'a') as f:
        #     f.write(f"Name: {name}, Score: {curScore}\n")
        if not name or not curScore:
            return jsonify({'error': 'Name and score are required'}), 400
       # Check if player with this name already exists
        existing_score = Score.query.filter_by(name=name).first()

        if existing_score:
            # Update the existing score
            existing_score.score = curScore
            db.session.commit()
            return jsonify({
                'success': True, 
                'message': 'Score updated successfully',
                'updated': True
            }), 200
        else:
            # Create new score record
            new_score = Score(name=name, score=curScore)
            db.session.add(new_score)
            db.session.commit()
            return jsonify({
                'success': True, 
                'message': 'New score saved successfully',
                'updated': False
            }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-scores', methods=['GET'])
def get_scores():
    try:
        scores = Score.query.order_by(Score.score.desc()).all()
        scores_list = [{'name': score.name, 'score': score.score} for score in scores]
        return jsonify({'success': True, 'scores': scores_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def init_db():
    with app.app_context():
        try:
            # Create tables using SQLAlchemy models
            db.create_all()
            
            # Execute the SQL commands directly
            db.session.execute("""
                CREATE TABLE IF NOT EXISTS Message (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL
                );
            """)
            
            db.session.execute("""
                CREATE TABLE IF NOT EXISTS Score (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    score INTEGER NOT NULL
                );
            """)
            
            db.session.commit()
            print("Database initialized successfully!")
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            db.session.rollback()


#if __name__ == '__main__':
#    print("\nServer running!")
#    print("WebSocket URL: http://localhost:8078/testsocket")
#    print("Click the URL above to open in your browser\n")
#    with app.app_context():
#        db.create_all()
#    socketio.run(app, host='0.0.0.0', port=8081)
    
# if __name__ == '__main__':
#    print("\nServer running!")
#    print("WebSocket URL: http://localhost:8078/testsocket")
#    print("Click the URL above to open in your browser\n")
#    with app.app_context():
#        db.create_all()
#    socketio.run(app, host='0.0.0.0', port=8081)
    
if __name__ == '__main__':
    with app.app_context():
        if not Path(f'{db_dir}/myacrobot.db').exists():
            print("Database file not found. Creating database...")
           # db.create_all()
            init_db()
            print("Database created successfully!")
        # db_path = Path('myacrobot.db')
        # if not db_path.exists():
        #     print("Database file not found. Creating database...")
        #     db.create_all()
        #     print("Database created successfully.")
#    with app.app_context():
#        db.create_all()
    socketio.run(app, host='0.0.0.0', port=443, ssl_context=('/app/certs/fullchain.pem', '/app/certs/privkey.pem'))

# if __name__ == '__main__':
#    socketio.run(app, host='0.0.0.0', port=80)

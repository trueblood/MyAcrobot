# app.py
from flask import Flask, request
# from app import routes

app = Flask(__name__)

# Initialize routes


if __name__ == "__main__":
    app.run(debug=True, port=9010)
    
    
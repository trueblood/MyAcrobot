# app/controllers/default_controller.py
from app.controllers import default_controller

from flask import render_template
from app import app

@app.route('/')
def index():
    print("in index")
    return render_template('index.html')


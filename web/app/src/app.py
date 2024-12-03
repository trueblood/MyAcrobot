from flask import Flask, render_template

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

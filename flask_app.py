import pickle
from flask import Flask,render_template,request

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('flask_index.html')

if __name__=='__main__':
    app.run(debug=True)
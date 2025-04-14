

import pickle
import re
import string
from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Function to check if file exists and load it
def load_pickle(file_name):
    if os.path.exists(file_name):
        return pickle.load(open(file_name, 'rb'))
    else:
        print(f"Error: {file_name} not found!")
        exit(1)  # Exit the script if the file is missing

# Load models and vectorizer
dtc_model = load_pickle('model_dtc.pkl')
gbc_model = load_pickle('model_gbc.pkl')
lr_model = load_pickle('model_lr.pkl')
rfc_model = load_pickle('model_rfc.pkl')
vectorization = load_pickle('vectorizer.pkl')  # Load vectorizer

# Text Preprocessing Function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

@app.route('/')
def index():
    return render_template('flask_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        processed_text = wordopt(news_text)
        new_x_test = [processed_text]

        # Transform the text using the loaded vectorizer
        new_xv_test = vectorization.transform(new_x_test)

        # Model Predictions
        pred_dtc = dtc_model.predict(new_xv_test)
        pred_gbc = gbc_model.predict(new_xv_test)
        pred_lr = lr_model.predict(new_xv_test)
        pred_rfc = rfc_model.predict(new_xv_test)

        # Map predictions to readable labels
        def output_label(n):
            return "It is Fake News" if n == 0 else "It is True News"

        predictions = {
            'Logistic Regression': output_label(pred_lr[0]),
            'Gradient Boosting': output_label(pred_gbc[0]),
            'Random Forest': output_label(pred_rfc[0]),
            'Decision Tree': output_label(pred_dtc[0]),
        }

        return render_template('flask_index.html', predictions=predictions, news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)


import os
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import ValidationError, InputRequired
from flask_wtf.file import FileAllowed 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


from dotenv import load_dotenv
load_dotenv()
#from config.py import IMAGE_SIZE

import requests


app = Flask(__name__)
app.config['SECRET_KEY'] ='dev'
app.config['UPLOAD_FOLDER'] = 'static/uploads'


# Load your trained model 
model = load_model('model/mammogram_model.h5')  # Make sure model.h5 is in your project folder
#print(model.summary())



class UploadImage(FlaskForm):   #using FlaskForm to create the html form and validate the input 
    file = FileField( validators=[InputRequired(), FileAllowed(['jpg','png','jpeg'])])
    submit = SubmitField('Upload')

@app.route('/', methods=["POST", "GET"])
def index():
    form = UploadImage()
    if form.validate_on_submit():
        file = form.file.data #first grab the file 
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        abs_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), save_path)

        print("Saving file to:", abs_path)
        file.save(abs_path)
      
    
        # Preprocess the image

        img = load_img(save_path, target_size=(256, 256))  # Ensure correct size
        img_array = img_to_array(img) / 255.0              # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension: (1, 256, 256, 3)
        
  


        # Predict
        prediction = model.predict(img_array)
        print("Raw model prediction:", prediction)

        class_names = ['birad1', 'birad3', 'birad4', 'birad5']  # ensuring order matches training labels
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]

        
        

        return render_template('result.html', prediction=predicted_class, image_path=save_path)

    return render_template('index.html', form = form)



#loading the llama chatbot
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

def query_groq_llama(user_message):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.5,
        "max_tokens": 512
    }

    print("Sending request to Groq API...")
    print("Payload:", payload)

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    print("Status code:", response.status_code)

    if response.status_code != 200:
        print("Groq API returned error response:")
        print(response.text)
        raise Exception(f"Groq API error: {response.text}")

    data = response.json()
    print("Response JSON:", data)

    # ✅ Extract the assistant's reply safely
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise Exception("Unexpected response format from Groq: 'choices[0].message.content' missing")
   

@app.route("/chat", methods=["POST","GET"])
def chat():
    return render_template ("chat.html")


@app.route("/response", methods=["POST"])
def response():

    print("📨 Received POST to /chat")  # 🔥 If you don’t see this, it’s not being called.

    data = request.get_json()
    user_input = data.get("message", "")

    print("User input:", user_input)

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        reply = query_groq_llama(user_input)
        print("Reply from LLaMA:", reply)
        return jsonify({"response": reply})
    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
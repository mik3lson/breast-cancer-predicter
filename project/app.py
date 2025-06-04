import os
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from dotenv import load_dotenv

from PIL import Image
from io import BytesIO
import uuid
import httpx


load_dotenv()

# FastAPI app
app = FastAPI()

# Constants
UPLOAD_FOLDER = "static/uploads/mammogram"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model('model/mammogram_model.h5')
class_names = ['birad1', 'birad3', 'birad4', 'birad5']

session_store = {}

@app.post("/predict/")
async def predict(
    patient_name: str = Form(...),
    patient_id:str =Form(...),
    file: UploadFile = File(...)
    ):
    # Save uploaded file
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Load and preprocess image
    img = load_img(file_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    # Grad-CAM
    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear

    score = CategoricalScore([predicted_index])
    gradcam = Gradcam(model, model_modifier=model_modifier, clone=True)
    cam = gradcam(score, img_array, penultimate_layer=-1)

    heatmap = cam[0]
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_img = np.uint8(img_array[0] * 255)

    if heatmap.shape != original_img.shape:
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    if original_img.dtype != np.uint8:
        original_img = original_img.astype(np.uint8)

    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    heatmap_path = os.path.join("static/uploads/heatmaps", "heatmap_" + filename)
    cv2.imwrite(heatmap_path, overlay)

    # Generate and store session_id
    session_id = str(uuid.uuid4())
    session_store[session_id] = {
        "patient_name": patient_name,
        "patient_id": patient_id,
        "predicted_class": predicted_class
    }

    return JSONResponse({
        "session_id": session_id,
        "patient_name": patient_name,
        "patient_id": patient_id,
        "predicted_class": predicted_class,
        "heatmap_path": heatmap_path,
        "uploaded_image_path": file_path
    })


#creating the medical report generator


@app.get("/generate-report/{session_id}")
async def generate_report(session_id: str):
    # 1. Check if session exists
    session = session_store.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found."}, status_code=404)

    patient_name = session["patient_name"]
    patient_id = session["patient_id"]
    predicted_class = session["predicted_class"]

    # 2. Format the prompt for the LLaMA model
    prompt = f"""
You are a medical AI generating a breast cancer report.

Patient Name: {patient_name}
Patient ID: {patient_id}
Diagnosis: {predicted_class.upper()}

Please write a short, professional medical summary for a radiologist.
"""

    # 3. Call Groq's LLaMA API 
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",  # Replace with actual Groq endpoint
                headers={
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": "You are a helpful medical report assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5
                }
            )
            data = response.json()
            report = data["choices"][0]["message"]["content"]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # 4. Return the report
    return JSONResponse({
        "patient_name": patient_name,
        "patient_id": patient_id,
        "predicted_class": predicted_class,
        "medical_report": report
    })





"""

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


#creating routes for the Chatbot

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

"""

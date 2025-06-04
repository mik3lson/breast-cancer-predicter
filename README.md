# Breast Cancer Mammogram Predictor API

This is a FastAPI backend that uses a CNN model to classify mammogram images into BI-RADS categories and provides explainability through Grad-CAM. It also features an LLM-powered medical report generator using the Groq API.

---

## 🚀 API Features

- 🧠 Predict BI-RADS category from a mammogram image
- 🔥 Visualize predictions using Grad-CAM heatmaps
- 📝 Generate AI-assisted radiology reports via LLaMA
- 📦 Fully compatible with Flutter or any frontend app

---

## 📂 Folder Structure

.
├── app.py # Main FastAPI app
├── model/ # Pretrained Keras model
├── static/uploads/ # Uploaded images & heatmaps
├── .env # Groq API key (if used)

## 📡 API Endpoints (for Flutter)

### 1. `POST /predict/`
Uploads a mammogram image and returns prediction, heatmap, and image paths.

**Request (multipart/form-data):**
- `file`: image file (jpg/jpeg/png)
- `patient_name`: string
- `patient_id`: string

**Response:**
```json
{
  "patient_name": "Jane Doe",
  "patient_id": "12345",
  "predicted_class": "birad4",
  "heatmap_path": "static/uploads/heatmaps/heatmap_image.png",
  "uploaded_image_path": "static/uploads/image.png",
  "session_id": "ae12-4bf0-99d2"  // use for report generation
}

##2. POST /generate-report/{session_id}
Generates a detailed medical report using Groq’s LLaMA model.

Request:

session_id: (provided in /predict/ response)

Response:

json

{
  "report": "Patient Jane Doe (ID: 12345) has a BI-RADS 4 mammogram..."
}
🛠️ Setup Instructions
1. Clone the repo:
git clone https://github.com/yourusername/breast-cancer-predictor-api.git
cd breast-cancer-predictor-api

2. Install dependencies:
pip install -r requirements.txt

3. Add your Groq API key to .env:
GROQ_API_KEY=your_key_here

4. Run the server:
uvicorn app:app --reload
5. Test at:
🔗 http://127.0.0.1:8000/docs — Swagger UI


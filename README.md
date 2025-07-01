# 🩺 Symptom Checker Chatbot

An intelligent medical assistant chatbot that predicts potential diseases and recommends appropriate medical specialists based on user-reported symptoms and patient profile data. Built with Natural Language Processing (NLP) and Machine Learning.

## 🚀 Features

- Accepts natural language input (e.g. "I have skin rash and fever").
- Maps symptoms and demographics to likely diseases using ML classifiers and/or embedding similarity search (FAISS).
- Recommends a medical specialist based on predicted disease.
- Designed for future deployment via WhatsApp using Twilio or similar.
- Trained on a real-world dataset of symptoms, diseases, and patient profiles.

## 🛠️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn / PyTorch or TensorFlow (depending on the model used)
- FAISS (Facebook AI Similarity Search)
- Hugging Face Transformers (for embeddings, if used)
- Flask / FastAPI (for API/chatbot backend)
- WhatsApp Integration (optional, via Twilio API)

## 💡 How It Works

1. The user inputs age, gender, and symptoms in natural language.
2. The system parses and maps symptoms using embeddings or rules.
3. The disease is predicted using a classifier or similarity matching.
4. The chatbot outputs:
   - Probable disease(s)
   - Recommended specialist
   - Optional follow-up or triage instructions

## ✅ Example Input

```json
{
  "age": 40,
  "gender": "Male",
  "symptoms": "I have skin rash with itching"
}

{
  "predicted_disease": "Allergic Dermatitis",
  "recommended_specialist": "Dermatologist"
}

git clone https://github.com/your-username/symptom-checker-chatbot.git
cd symptom-checker-chatbot
pip install -r requirements.txt
python main.py

📲 Deployment
For WhatsApp deployment: integrate the backend with the Twilio API.

Optionally containerize with Docker for cloud deployment.

🧪 Future Work
Expand to multilingual support.

Integrate with electronic health records (EHR).

Fine-tune disease prediction with larger datasets and LLMs.

Build conversation memory and follow-up logic.
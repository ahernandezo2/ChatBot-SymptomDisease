from fastapi import FastAPI, Form
from pydantic import BaseModel
from app.logic import get_diagnosis
from twilio.twiml.messaging_response import MessagingResponse
import re

app = FastAPI()

class SymptomQuery(BaseModel):
    symptoms: str
    age: int
    gender: str

@app.post("/diagnose")
def diagnose(input: SymptomQuery):
    response_text = get_diagnosis(input.symptoms, input.age, input.gender)
    return {"response": response_text}

@app.post("/webhook")
def whatsapp_webhook(Body: str = Form(), From: str = Form()):
    """Handle incoming WhatsApp messages"""
    response = MessagingResponse()
    message = Body.strip()
    
    if message.lower() in ["help", "start", "hi", "hello"]:
        response.message("👋 Hi! I can help identify possible medical conditions.\n\nPlease tell me:\n• Your symptoms\n• Your age\n• Your gender\n\nExample: 'I have fever and cough, I'm 25, male'")
    else:
        try:
            # Extract age and gender
            age_match = re.search(r'\b(\d{1,3})\b', message)
            gender_match = re.search(r'\b(male|female|m|f)\b', message, re.IGNORECASE)
            
            if age_match and gender_match:
                age = int(age_match.group(1))
                gender = "Male" if gender_match.group(1).lower() in ['male', 'm'] else "Female"
                
                # Get diagnosis
                diagnosis = get_diagnosis(message, age, gender)
                response.message(f"{diagnosis}\n\n⚠️ *Disclaimer:* This is for information only. Please consult a healthcare professional.")
            else:
                response.message("Please include your age and gender.\nExample: 'I have fever and cough, I'm 25, male'")
                
        except Exception as e:
            response.message("Sorry, something went wrong. Please try again or type 'help'.")
    
    return str(response)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://spam-frontend-one.vercel.app"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("spam_model.pkl")


class EmailRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Spam Detection API Running"}


@app.post("/predict")
def predict(data: EmailRequest):
    prediction = model.predict([data.text])[0]
    prob = model.predict_proba([data.text])[0]

    status = "spam" if prediction == 1 else "safe"
    percentage = round(max(prob) * 100)

    risks = []
    urls = re.findall(r'https?://\S+', data.text)

    text = data.text.lower()

    if urls:
        risks.append("Contains clickable links")

    if "urgent" in text:
        risks.append("Urgency-related phrases detected")

    if "verify" in text:
        risks.append("Requests account verification")

    if "login" in text:
        risks.append("Contains login-related wording")

    return {
        "status": status,
        "percentage": percentage,
        "risks": risks,
        "urls": urls
    }
    
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

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

    return {
        "status": "spam" if prediction == 1 else "safe",
        "percentage": round(prob[1] * 100)
    }
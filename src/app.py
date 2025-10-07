from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
model = pipeline("text-classification", model="../models/fake-news-model")

@app.get("/")
def home():
    return {"message": "Fake News Classifier API running 🚀"}

@app.post("/predict/")
def predict(text: str):
    result = model(text)
    return {"prediction": result}
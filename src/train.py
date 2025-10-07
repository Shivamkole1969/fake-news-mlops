from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import pipeline
import pandas as pd

def train():
    # Fake dataset
    data = {
        "text": [
            "Breaking: NASA finds water on Mars!",
            "Celebrity says aliens built the pyramids!",
            "COVID-19 vaccine causes superpowers!",
            "Stock markets rise after tech rally",
        ],
        "label": [0, 1, 1, 0],  # 0 = real, 1 = fake
    }

    df = pd.DataFrame(data)

    # Using a small pre-trained transformer
    clf = pipeline("text-classification", model="distilbert-base-uncased")

    # Save model (dummy save)
    clf.save_pretrained("../models/fake-news-model")

    print("✅ Model trained and saved at models/fake-news-model")

if __name__ == "__main__":
    train()
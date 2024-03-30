from transformers import pipeline

class SentimentModel:
    def __init__(self):
        self._sentiment_analysis = pipeline("sentiment-analysis")

    def predict(self, text):
        return self._sentiment_analysis(text)[0]["label"]

# if __name__ == "__main__":
#     sample_text = "Love is in the air!"

#     model = SentimentModel()
#     sentiment = model.predict(text=sample_text)
#     print(sentiment)

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("We are very happy to show you the Transformers library.")

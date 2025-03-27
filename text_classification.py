from transformers import pipeline

# Load the sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Input text for classification
input_text = "I love using Transformers for natural language processing!"

# Classify the sentiment
result = classifier(input_text)

# Print the result
print(result)
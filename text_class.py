# Step 1: Import the necessary module
from transformers import pipeline

# Step 2: Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Step 3: Define your texts and labels
texts = [
    "Djokovic has qualified for his sixth French Open final!",
    "On March 23, 2010, President Obama signed the Affordable Care Act into law.",
    "The goal was to show that scientists from various disciplines could find common ground about climate action."
]

labels = ["Sport", "Politics", "Environment"]

# Step 4: Classify the texts
for text in texts:
    result = classifier(text, labels)
    print(f"Text: {text}")
    print("Predicted Labels:")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"- {label}: {score:.4f}")
    print()
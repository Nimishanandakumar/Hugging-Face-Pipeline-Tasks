from transformers import pipeline

# Initialize the NER pipeline
ner_pipeline = pipeline("ner", aggregation_strategy="simple")

# Sample text for NER
text = "Barack Obama was the 44th President of the United States."

# Perform NER
entities = ner_pipeline(text)

# Print the identified entities
for entity in entities:
    print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.4f}")
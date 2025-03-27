from transformers import pipeline

# Initialize the masked language modeling pipeline
masked_lm = pipeline("fill-mask", model="roberta-base")  # You can also use "bert-base-uncased"

# Use the correct mask token
masked_result = masked_lm("The capital of France is <mask>.")

# Print the predictions
print("Masked Language Modeling:")
for result in masked_result:
    print(f"Prediction: {result['token_str']}, Score: {result['score']:.4f}")
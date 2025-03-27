from transformers import pipeline

# Initialize the text generation pipeline with GPT-2
text_generator = pipeline("text-generation", model="gpt2")

# Define a prompt for text generation
prompt = "In a future world where technology has advanced beyond our imagination,"

# Generate text
generated_text = text_generator(prompt, max_length=50, num_return_sequences=1, truncation = True)

# Print the generated text
print(generated_text[0]['generated_text'])
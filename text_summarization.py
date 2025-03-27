from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Sample long text
long_text = """
The Transformers library by Hugging Face provides a simple and efficient way to work with state-of-the-art natural language processing models. 
It allows users to easily load pre-trained models and perform various tasks such as text classification, named entity recognition, and text generation. 
The library is built on top of PyTorch and TensorFlow, making it flexible and powerful for both research and production use cases.
"""

# Generate a summary
summary = summarizer(long_text, max_length=50, min_length=25, do_sample=False)

# Print the summary
print(summary[0]['summary_text'])
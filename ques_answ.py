question_answerer = pipeline("question-answering")
qa_result = question_answerer(question="What is Hugging Face?", context="Hugging Face is a company that provides tools for natural language processing.")
print("Question Answering:")
print(qa_result)
print()
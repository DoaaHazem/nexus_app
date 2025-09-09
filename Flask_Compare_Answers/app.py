# Import necessary libraries
from flask import Flask, request, jsonify  
from transformers import AutoTokenizer, AutoModel  
from sklearn.metrics.pairwise import cosine_similarity  
import torch  
import os  

# Initialize Flask application
app = Flask(__name__)

# Load the tokenizer and model once when the server starts
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Load tokenizer
model = AutoModel.from_pretrained('bert-base-uncased')  # Load model

# Function to compute the embedding of a given text
def get_embedding(text):
    # Tokenize input text and convert to tensor format
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Run the model in inference mode (no gradients needed)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Return the mean of the token embeddings as the sentence embedding
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Define an API endpoint for comparing answers
@app.route('/compare', methods=['POST'])
def compare():
    # Check if the file was provided in the request
    if 'answers_file' not in request.files:
        return jsonify({'error': 'answers_file is required'}), 400

    # Get uploaded file and form data
    answers_file = request.files['answers_file']
    question_number = int(request.form.get('question_number', 0))  # Question number from form
    student_answer = request.form.get('student_answer', '')  # Student's answer from form

    # Validate question number and student answer
    if not student_answer or question_number <= 0:
        return jsonify({'error': 'Invalid question_number or student_answer'}), 400

    # Read lines from the uploaded file
    answers_lines = answers_file.read().decode('utf-8').splitlines()
    
    # Check if the question number is within range
    if question_number > len(answers_lines):
        return jsonify({'error': 'Question number exceeds number of lines in file'}), 400

    # Extract the correct answer for the given question number
    correct_answer = answers_lines[question_number - 1].strip()

    # Generate embeddings for both the correct and student answers
    correct_embedding = get_embedding(correct_answer)
    student_embedding = get_embedding(student_answer)

    # Compute cosine similarity between embeddings
    similarity = cosine_similarity(correct_embedding, student_embedding)[0][0]

    # Return result with similarity score and correct answer
    return jsonify({
        'result': "Correct" if similarity >= 0.6 else "Incorrect",  # Threshold for correctness
        'similarity_score': float(similarity),  # Raw similarity score
        'correct_answer': correct_answer  # Return correct answer for reference
    })

# Run the app in debug mode if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)

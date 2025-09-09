from flask import Flask, request, send_file, jsonify, after_this_request
import tempfile, os, re
from zipfile import ZipFile
from datetime import datetime
import time
from pydub import AudioSegment
import whisper
import subprocess
import glob
import shutil
from transformers import (
    AutoTokenizer,                      
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import pandas as pd
import torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
# =================== Save Files =====================

UPLOAD_FOLDER = 'saved_files'      # Folder to save uploaded files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================== Load Models ====================

asr_model = whisper.load_model("large")

t5_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-e2e-qg",legacy=False)
t5_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-e2e-qg")

qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

user_key ="API-KEY"
llm = ChatGroq(
    groq_api_key=user_key,
    model_name="deepseek-r1-distill-llama-70b"
)

prompt_template = """
Analyze the following text {text} and perform the following tasks in English, regardless of the input language:

1. Generate an indirect, complex, and analytical question that requires interpretation or inference based on the text. Ensure that:
   - The question is clear and sufficiently complex.
   - Vague or ambiguous phrases like 'in the text' are avoided.
   - No part of the question implicitly answers.

2. Provide a direct, concise answer to the generated question based solely on the content and information presented in the text.

Each task should be distinct, and the answer must not repeat or paraphrase the question itself.

Format your response as follows:
- Begin the response with the word 'Question:' followed by the question content.
- Use the word 'Answer:' to introduce the concise response."
"""
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

# ==================== Functions ====================

def transcribe_video_with_openai(video_path):
   #1. Extract audio from video (to a temporary WAV file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_output = temp_audio.name

    try:
        # Extract audio using ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,  # -y: overwrite output, -i: input file
            "-vn",                             # Skip video stream
            "-acodec", "pcm_s16le",            # Use WAV-compatible audio codec
            "-ar", "16000",                    # Set sample rate to 16 kHz
            "-ac", "1",                        # Mono audio
            audio_output
        ], check=True)

        #2. Create a temporary folder to split the audio.
        temp_dir = tempfile.mkdtemp()

        # Cut audio into 5 minute (300 second) parts
        segment_pattern = os.path.join(temp_dir, "segment_%03d.wav")
        subprocess.run([
            "ffmpeg", "-i", audio_output,
            "-f", "segment",                  # Enable segmenting mode
            "-segment_time", "300",           # 300 seconds = 5 minutes
            "-c", "copy", segment_pattern     # Copy codec (no re-encoding)
        ], check=True)

        #3. Collect all the cut audio files
        audio_files = sorted(glob.glob(os.path.join(temp_dir, "segment_*.wav")))

        # Initialize an empty string to hold the final transcription 
        full_transcription = ""

        # Transcribe each audio segment using OpenAI Whisper API
        for i, audio_file in enumerate(audio_files):
            print(f"Transcribing segment {i+1}/{len(audio_files)}: {audio_file}")

            with open(audio_file, "rb") as f:
                try:
                    transcript = openai.Audio.transcribe("whisper-1", f)
                    full_transcription += transcript["text"] + "\n"
                except Exception as e:
                    print(f"Error transcribing segment {audio_file}: {e}")
                    full_transcription += f"\n[خطأ في المقطع {i+1}]\n"

    finally:
       # Clean up temporary audio file and folder after processing
        if os.path.exists(audio_output):
            os.remove(audio_output)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return full_transcription  # Return the full combined transcription


def summarize_text_English(text, llm):
    prompt = (f"Summarize the following text efficiently, providing only the existing key insights in English without adding any additional information: \n{text}"
              f"Highlight the most critical points."
              f"Begin the response with the word 'Summary' followed by the summary content, and use the phrase 'Critical Points' to start the list of critical points")
    response = llm.invoke(prompt)
    raw_summary = response.content if hasattr(response, 'content') else response
    cleaned_summary_en = re.sub(r"<think>.*?</think>", "", raw_summary, flags=re.DOTALL).strip()
    return cleaned_summary_en

def summarize_text_Arabic(text, llm):
    prompt = (f"Summarize the following text in Egyptian Arabic while preserving English terms as they are, without adding any additional information: \n{text}"
              f"Highlight the most critical points in Egyptian Arabic."
              f"Begin the response with the word 'Summary' followed by the summary content, and use the phrase 'Critical Points' to start the list of critical points")
    response = llm.invoke(prompt)
    raw_summary = response.content if hasattr(response, 'content') else response
    cleaned_summary_ar = re.sub(r"<think>.*?</think>", "", raw_summary, flags=re.DOTALL).strip()
    return cleaned_summary_ar

def separate_summary_and_critical_points(cleaned_summary_en,cleaned_summary_ar):
    summary_match_en = re.search(r"Summary(.*?)(Critical(.*)|$)", cleaned_summary_en, re.DOTALL)
    critical_points_match_en = re.search(r"Critical(.*)", cleaned_summary_en, re.DOTALL)

    summary_match_ar = re.search(r"Summary(.*?)(Critical(.*)|$)", cleaned_summary_ar, re.DOTALL)
    critical_points_match_ar = re.search(r"Critical(.*)", cleaned_summary_ar, re.DOTALL)

    Summary_English = (summary_match_en.group(1).strip() + "\n") if summary_match_en else ""
    Notes_English = critical_points_match_en.group(1).strip() if critical_points_match_en else ""

    Summary_Arabic = (summary_match_ar.group(1).strip() + "\n") if summary_match_ar else ""
    Notes_Arabic = critical_points_match_ar.group(1).strip() if critical_points_match_ar else ""

    return Summary_English, Summary_Arabic, Notes_English, Notes_Arabic

def simple_questions(text):
    input_text = f"generate question: {text} </s>"
    inputs = t5_tokenizer(
        input_text, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=512)
    outputs = t5_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=64,
        num_beams=5,
        num_return_sequences=1,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3
    )
    question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def extract_answers(context, question):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt")
    outputs = qa_model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1
    answer_tokens = inputs['input_ids'][0][start_index:end_index]
    answer = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

def complex_questions_answers(input_text):
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(text=input_text)
    cleaned_questions_answers = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return cleaned_questions_answers

def separate_Questions_and_Answers_complex(text):
    question = re.search(r"Question(.*?)(Answer(.*)|$)", text, re.DOTALL)
    Answer = re.search(r"Answer(.*)", text, re.DOTALL)

    # Extract summary and critical points or fallback to empty strings
    question = (question.group(1).strip() + "\n") if question else ""
    lines = question.splitlines()
    # Remove lines that contain only '**'
    cleaned_lines = [line for line in lines if line.strip() != '**']
    # Join the lines back into a single string
    question = "\n".join(cleaned_lines)
    
    Answer = Answer.group(1).strip() if Answer else ""

    return question, Answer

def get_latest_transcribed_text():
    files = sorted(glob.glob(os.path.join(UPLOAD_FOLDER, "*_transcribed.txt")), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        return f.read()

def get_latest_summary_en():
    files = sorted(glob.glob(os.path.join(UPLOAD_FOLDER, "summary_*_english.txt")), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        return f.read()

# ==================== Endpoints ====================
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return "No video file provided", 400

    file = request.files['file']
    # Generate a timestamped base filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{os.path.splitext(file.filename)[0]}_{timestamp}"
    video_filename = os.path.join(UPLOAD_FOLDER, f"{base_filename}{os.path.splitext(file.filename)[1]}")
    file.save(video_filename) # Save uploaded video to the upload folder

    try:  # Transcribe the video using OpenAI Whisper and get the full text
        text = transcribe_video_with_openai(video_filename)
        text_path = os.path.join(UPLOAD_FOLDER, f"{base_filename}_transcribed.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

    except Exception as e: # On error, delete the uploaded video file and return error response
        if os.path.exists(video_filename):
            os.remove(video_filename)
        return f"Error processing video: {e}", 500

    # After transcription is saved, delete the original video to save space
    if os.path.exists(video_filename):
        os.remove(video_filename)

    # Send the transcribed text file back to the user as a download
    return send_file(text_path, as_attachment=True, download_name="transcribed_text.txt")

@app.route('/summaries', methods=['POST'])
def generate_summaries():
    text = get_latest_transcribed_text()  # Load the most recent transcribed text file
    if not text:
        return "No transcription available", 400

    # Generate English and Arabic summaries (ignore critical points for now)
    summary_en, summary_ar, _, _ = separate_summary_and_critical_points(
        summarize_text_English(text, llm),
        summarize_text_Arabic(text, llm)
    )
    # Create a unique timestamped base filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"summary_{timestamp}"
    summary_path_en = os.path.join(UPLOAD_FOLDER, f"{base_filename}_english.txt")
    summary_path_ar = os.path.join(UPLOAD_FOLDER, f"{base_filename}_arabic.txt")
    zip_path = os.path.join(UPLOAD_FOLDER, f"{base_filename}.zip")
    # Write English summary to file
    with open(summary_path_en, "w", encoding="utf-8") as f_en:
        f_en.write(summary_en)
    # Write Arabic summary to file
    with open(summary_path_ar, "w", encoding="utf-8") as f_ar:
        f_ar.write(summary_ar)
    # Compress both summaries into a single ZIP file
    with ZipFile(zip_path, 'w') as zipf:
        zipf.write(summary_path_en, arcname="summary_english.txt")
        zipf.write(summary_path_ar, arcname="summary_arabic.txt")

    return send_file(zip_path, as_attachment=True, download_name="summaries.zip")

@app.route('/notes', methods=['POST'])
def generate_notes():
    text = get_latest_transcribed_text() # Retrieve the latest available transcribed text
    if not text:
        return "No transcription available", 400
    
    # Generate critical notes in English and Arabic
    # We ignore summaries (first two outputs), keep only the notes
    _, _, notes_en, notes_ar = separate_summary_and_critical_points(
        summarize_text_English(text, llm),
        summarize_text_Arabic(text, llm)
    )
    # Generate a unique timestamped base name for file saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"notes_{timestamp}"
    notes_path_en = os.path.join(UPLOAD_FOLDER, f"{base_filename}_english.txt")
    notes_path_ar = os.path.join(UPLOAD_FOLDER, f"{base_filename}_arabic.txt")
    zip_path = os.path.join(UPLOAD_FOLDER, f"{base_filename}.zip")
    # Save English notes to file
    with open(notes_path_en, "w", encoding="utf-8") as f_en:
        f_en.write(notes_en)
    # Save Arabic notes to file
    with open(notes_path_ar, "w", encoding="utf-8") as f_ar:
        f_ar.write(notes_ar)
    # Create a ZIP file containing both English and Arabic notes
    with ZipFile(zip_path, 'w') as zipf:
        zipf.write(notes_path_en, arcname="notes_english.txt")
        zipf.write(notes_path_ar, arcname="notes_arabic.txt")
    # Automatically remove temporary note files after the response is sent
    @after_this_request
    def remove_files(response):
        try:
            os.remove(notes_path_en)
            os.remove(notes_path_ar)
            os.remove(zip_path)
        except Exception as e:
            print(f"Error deleting temp files: {e}")
        return response

    return send_file(zip_path, as_attachment=True, download_name="notes.zip")

@app.route('/questions_answers', methods=['POST'])
def generate_questions_answers(): 
    # Get the latest transcription and English summary
    text = get_latest_transcribed_text()
    summary_en = get_latest_summary_en()

    if not text or not summary_en:
        return "Transcription or summary not available", 400

    questions = []
    answers = []
    idx = 1   # Counter for numbering questions and answers

    # ====== Generate Complex Questions and Answers ======
    complex_qas_list = complex_questions_answers(text)
    for qas_text in complex_qas_list:
        question, answer = separate_Questions_and_Answers_complex(qas_text)
        if question:
            questions.append(f"{idx}. {question}")
            answers.append(f"{idx}. {answer}")
            idx += 1
    # ====== Generate Simple Questions and Answers ======
    # Generate simple questions based on the English summary
    simple_q_text = simple_questions(summary_en)
    simple_q_list = simple_q_text.split('<sep>')
    simple_q_list = [q.strip() for q in simple_q_list if q.strip()]

    for q in simple_q_list: # For each simple question, extract the answer from the original text
        a = extract_answers(text, q)
        if a:
            questions.append(f"{idx}. {q}")
            answers.append(f"{idx}. {a}")
            idx += 1

    # ====== Save Questions and Answers to Excel Files ======
    q_df = pd.DataFrame({"Question": questions})
    a_df = pd.DataFrame({"Answer": answers})

    # Timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"qa_{timestamp}"
    questions_path = os.path.join(UPLOAD_FOLDER, f"{base_filename}_questions.xlsx")
    answers_path = os.path.join(UPLOAD_FOLDER, f"{base_filename}_answers.xlsx")
    zip_path = os.path.join(UPLOAD_FOLDER, f"{base_filename}.zip")

    # Save Excel files
    q_df.to_excel(questions_path, index=False)
    a_df.to_excel(answers_path, index=False)

    # ====== Compress Excel Files into a ZIP Archive ======
    with ZipFile(zip_path, 'w') as zipf:
        zipf.write(questions_path, arcname="questions.xlsx")
        zipf.write(answers_path, arcname="answers.xlsx")

    # ====== Delete the ZIP file after sending (Excel files remain) ======
    @after_this_request
    def remove_zip(response):
        try:
            os.remove(zip_path)  # remove zip only; keep source files
        except Exception as e:
            print(f"Error deleting zip file: {e}")
        return response

    return send_file(zip_path, as_attachment=True, download_name="questions_and_answers.zip")

# ==================== Run App ====================
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)  

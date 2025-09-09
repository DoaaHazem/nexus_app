# Import Libraries
from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
import mediapipe as mp
import time
from ultralytics import YOLO

app = Flask(__name__)

# Load The Models
face_model = YOLO("yolov8n (1).pt")  
mobile_model = YOLO("best best (mobile) .pt")  
face_mesh = mp.solutions.face_mesh.FaceMesh()

# Dictionary to store data for each session
session_data = {}

def calculate_attention(frame, session_id):
    if session_id not in session_data:
        session_data[session_id] = {
            "start_time": time.time(),
            "sleep_time": 0,
            "no_attendance_time": 0,
            "people_time": 0,
            "last_seen": time.time(),
            "people_count": 0,
            "mobile_detected": False
        }

    # Detect faces and mobiles in the frame
    results = face_model(frame)
    mobile_results = mobile_model(frame)

    # Calculate the number of people and the number of mobiles in the frame
    people_count = sum(len(r.boxes) for r in results)  
    mobile_detected = any(len(r.boxes) > 0 for r in mobile_results) 

    current_time = time.time() # Calculate the current time

    # Update attendance times, absence times and number of people
    if people_count == 0:
        session_data[session_id]["no_attendance_time"] += current_time - session_data[session_id]["last_seen"]
    elif people_count > 1:
        session_data[session_id]["people_time"] += current_time - session_data[session_id]["last_seen"]

    # Update current session data
    session_data[session_id]["last_seen"] = current_time
    session_data[session_id]["people_count"] = people_count
    session_data[session_id]["mobile_detected"] = mobile_detected

    # Calculate the initial attention
    total_score = 100
    if session_data[session_id]["no_attendance_time"] > 10: # Discount if the absence exceeds 10 seconds.
        total_score -= 20
    if session_data[session_id]["mobile_detected"]: # Discount if the mobile is monitored
        total_score -= 30
    if session_data[session_id]["sleep_time"] > 15: # Discount if sleep time exceeds 15 seconds
        total_score -= 20

    total_score = max(0, total_score) # Check if the total_score is not negative
    percentage_of_attention = (total_score / 100) * 100 # Calculate the final attention ratio

    return {
        "Percentage_of_attention": f"{percentage_of_attention}%",
        "Mobile": session_data[session_id]["mobile_detected"],
        "Sleep_time": session_data[session_id]["sleep_time"],
        "No_attendance_time": session_data[session_id]["no_attendance_time"],
        "People_count": session_data[session_id]["people_count"],
        "People_time": session_data[session_id]["people_time"]
    }

def calculate_cheating(frame, session_id):
    if session_id not in session_data:
        session_data[session_id] = {
            "start_time": time.time(),
            "sleep_time": 0,
            "no_attendance_time": 0,
            "people_time": 0,
            "last_seen": time.time(),
            "people_count": 0,
            "mobile_detected": False
        }

    # Detect faces and mobiles in the frame
    results = face_model(frame)
    mobile_results = mobile_model(frame)
    
    # Calculate the number of people and the number of mobiles in the frame
    people_count = sum(len(r.boxes) for r in results)  
    mobile_detected = any(len(r.boxes) > 0 for r in mobile_results)  

    current_time = time.time() # Calculate the current time
    
    # Update attendance times, absence times and number of people
    if people_count == 0:
        session_data[session_id]["no_attendance_time"] += current_time - session_data[session_id]["last_seen"]
    elif people_count > 1:
        session_data[session_id]["people_time"] += current_time - session_data[session_id]["last_seen"]

    # Update current session data
    session_data[session_id]["last_seen"] = current_time
    session_data[session_id]["people_count"] = people_count
    session_data[session_id]["mobile_detected"] = mobile_detected

    # Initial cheating score calculation
    total_score = 100
    if session_data[session_id]["no_attendance_time"] > 10: # If the absence exceeds 10 seconds, it is considered complete cheating.
        total_score = 0  
    if session_data[session_id]["mobile_detected"]: # 30% discount if you have a mobile phone
        total_score -= 30

    total_score = max(0, total_score) # Check if the total_score is not negative
    percentage_of_cheating = (100 - total_score) # Cheating percentage calculation

    return {
        "Percentage_of_cheating": f"{percentage_of_cheating}%",
        "Mobile": session_data[session_id]["mobile_detected"],
        "Sleep_time": session_data[session_id]["sleep_time"],
        "No_attendance_time": session_data[session_id]["no_attendance_time"],
        "People_count": session_data[session_id]["people_count"],
        "People_time": session_data[session_id]["people_time"]
    }

@app.route('/attention', methods=['POST'])
def attention():
    file = request.files['image'].read()
    session_id = request.form.get("session_id", "default")
    np_img = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    result = calculate_attention(img, session_id)
    return jsonify(result)

@app.route('/cheating', methods=['POST'])
def cheating():
    file = request.files['image'].read()
    session_id = request.form.get("session_id", "default")
    np_img = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    result = calculate_cheating(img, session_id)
    return jsonify(result)

# To run the application locally (suspend usage time with gunicorn)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

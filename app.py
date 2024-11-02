from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import base64
import io
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app)

KNOWN_FACES_DIR = 'known_faces'
known_face_encodings = []
known_face_names = []

# Function to reset attendance records
def reset_attendance():
    attendance_df = pd.DataFrame(columns=["Name", "Time"])
    attendance_df.to_csv("attendance.csv", index=False)
    print("Attendance file has been reset.")

# Function to load known faces
def load_known_faces():
    known_face_encodings.clear()
    known_face_names.clear()
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)
                embedding = DeepFace.represent(image_path, model_name='VGG-Face')[0]["embedding"]
                known_face_encodings.append(embedding)
                known_face_names.append(person_name)
    print("Finished loading known faces.")

# Function to mark attendance
def mark_attendance(name):
    attendance_df = pd.read_csv("attendance.csv")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Mark attendance only if it's not already recorded for today
    if not ((attendance_df['Name'] == name) & (attendance_df['Time'].str[:10] == current_time[:10])).any():
        new_entry = pd.DataFrame([[name, current_time]], columns=["Name", "Time"])
        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
        attendance_df.to_csv("attendance.csv", index=False)
        # Send the updated attendance list to the client
        socketio.emit('attendance_update', {'name': name, 'time': current_time})

# Handle video frame from client
@socketio.on('video_frame')
def handle_video_frame(data):
    # Decode the base64 image received from the client
    image_data = base64.b64decode(data.split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Detect faces and generate embeddings
    faces = DeepFace.extract_faces(img, detector_backend="opencv", enforce_detection=False)
    
    if faces:
        face_img = faces[0]["face"]  # Extract the first face from the image
        embedding = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
        
        # Compare the embedding with known faces
        most_similar_name = "Unknown"
        highest_similarity = 0.0

        for idx, known_encoding in enumerate(known_face_encodings):
            similarity = cosine_similarity([embedding], [known_encoding])[0][0]
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_name = known_face_names[idx]

        # Mark attendance if similarity is high enough
        if most_similar_name != "Unknown" and highest_similarity > 0.25:
            mark_attendance(most_similar_name)

# Route to display the main attendance system
@app.route('/')
def home():
    return render_template('index.html')

# Route to upload known faces
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            name = request.form['name']
            person_dir = os.path.join(KNOWN_FACES_DIR, name)
            os.makedirs(person_dir, exist_ok=True)

            # Handle uploaded files
            for file in request.files.getlist('files'):
                if file and file.filename.endswith(('jpg', 'jpeg', 'png')):
                    file.save(os.path.join(person_dir, file.filename))

            # Handle multiple captured images
            captured_images = request.form.getlist('captured_images[]')
            for idx, captured_image_data in enumerate(captured_images):
                if captured_image_data and captured_image_data.startswith('data:image'):
                    image_data = base64.b64decode(captured_image_data.split(',')[1])
                    image = Image.open(io.BytesIO(image_data))
                    image.save(os.path.join(person_dir, f'{name}_captured_{idx}.jpg'))

            load_known_faces()
            return "Images uploaded successfully!"
        except Exception as e:
            return f"Error uploading images: {str(e)}", 400
    return render_template('upload.html')

if __name__ == '__main__':
    reset_attendance()
    load_known_faces()  # Load faces initially
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
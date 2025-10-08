# Face Recognition WebApp

A Flask-based web application for **face recognition** from images, videos, and live webcam feeds.  
Built with [face_recognition](https://github.com/ageitgey/face_recognition) (dlib + OpenCV) and a Bootstrap frontend.

---

## Features

- Upload an **image** → detect and identify known people  
- Upload a **video** → extract snapshots of recognized faces with timestamps  
- Use your **webcam** for live recognition  
- Add new people to the **known database** via the web interface  
- History of recent detections is stored per session  
- Reference images for matches are displayed for verification  

---

## Project Structure

face_webapp/
│
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates (Flask Jinja2)
│   ├── index.html
│   ├── result.html
│   └── webcam.html
├── static/               # Static files
│   └── snapshots/        # Snapshots from video recognition
├── uploads/              # Uploaded files (images/videos)
├── known/                # Database of known people (subfolders per person)
└── venv/                 # Virtual environment (ignored in git)

---

## Installation (Ubuntu / Linux)

1. Clone the repo:
   ```bash
   git clone git@github.com:Jordon1634/face_webapp.git
   cd face_webapp

2. Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

3. Install system dependencies:
sudo apt update
sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev libboost-all-dev libgl1 ffmpeg

4. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

---

##Usage

Start the app
python app.py

Open in your browser
http://127.0.0.1:5000

1. Upload an Image
	•	Click Picture Upload
	•	Faces will be detected and matched against your known database
	•	Results show “Original” vs “Processed” image + matched reference images

2. Upload a Video
	•	Click Video Upload
	•	App processes frames at ~5 fps
	•	Snapshots of recognized faces are saved with timestamps under static/snapshots/

3. Live Webcam
	•	Click Live Webcam
	•	Opens your system webcam feed
	•	Recognized people are labeled in real time

4. Add New People
	•	Use Add a New Person
	•	Provide a name and upload a face photo
	•	A folder will be created under known/<PersonName>/ to store images
	•	The app automatically rebuilds encodings when new people are added

---

##Notes

•	Empty folders (known/, uploads/, static/, etc.) are kept with .gitkeep so the app runs out of the box.
	•	Encodings are cached in encodings.pkl for faster reloads. Delete this file if you update the known/ folder manually.
	•	By default, the face recognition model uses HOG. For better accuracy (but slower), you can change it to cnn in the code if CUDA is available.

---

##License

This project is for educational and training purposes.
Use responsibly.
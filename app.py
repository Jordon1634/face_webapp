import os
import io
import cv2
import uuid
import pickle
from datetime import timedelta
from typing import List, Tuple, Dict

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash, Response
from werkzeug.utils import secure_filename

import numpy as np
import face_recognition

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
KNOWN_FOLDER = os.path.join(APP_ROOT, 'known')
SNAPSHOT_FOLDER = os.path.join(APP_ROOT, 'static', 'snapshots')
ENCODE_CACHE = os.path.join(APP_ROOT, 'encodings.pkl')

ALLOWED_IMAGE = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO = {'mp4', 'mov', 'avi', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'devkey')

# ---------- Helper utils ----------

def allowed_file(filename, exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts

def load_known_faces() -> Tuple[List[np.ndarray], List[str], Dict[str, str]]:
    cache = None
    if os.path.exists(ENCODE_CACHE):
        try:
            with open(ENCODE_CACHE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            cache = None

    latest_mtime = 0
    for root, _, files in os.walk(KNOWN_FOLDER):
        for fn in files:
            if allowed_file(fn, ALLOWED_IMAGE):
                latest_mtime = max(latest_mtime, os.path.getmtime(os.path.join(root, fn)))

    if cache and cache.get('mtime', 0) >= latest_mtime:
        return cache['encodings'], cache['names'], cache['rep_image']

    encodings, names = [], []
    rep_image: Dict[str, str] = {}
    for person in sorted(os.listdir(KNOWN_FOLDER)):
        pdir = os.path.join(KNOWN_FOLDER, person)
        if not os.path.isdir(pdir):
            continue
        imgs = [f for f in os.listdir(pdir) if allowed_file(f, ALLOWED_IMAGE)]
        for f in imgs:
            path = os.path.join(pdir, f)
            img = face_recognition.load_image_file(path)
            locs = face_recognition.face_locations(img, model='hog')
            if not locs:
                continue
            encs = face_recognition.face_encodings(img, locs)
            for enc in encs:
                encodings.append(enc)
                names.append(person)
                rep_image.setdefault(person, path.replace(APP_ROOT + os.sep, ''))
    try:
        with open(ENCODE_CACHE, 'wb') as f:
            pickle.dump({'encodings': encodings, 'names': names, 'rep_image': rep_image, 'mtime': latest_mtime}, f)
    except Exception:
        pass
    return encodings, names, rep_image

def annotate_image(image_bgr, boxes, labels):
    for (top, right, bottom, left), label in zip(boxes, labels):
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(image_bgr, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return image_bgr

def ensure_history():
    if 'history' not in session:
        session['history'] = []

def add_history(entry):
    ensure_history()
    session['history'] = session.get('history', [])
    session['history'].insert(0, entry)
    session.modified = True

# ---------- Routes ----------

@app.route('/')
def index():
    ensure_history()
    history = session.get('history', [])
    known_people = sorted([d for d in os.listdir(KNOWN_FOLDER) if os.path.isdir(os.path.join(KNOWN_FOLDER, d))])
    return render_template('index.html', history=history, known_people=known_people)

@app.route('/add_person', methods=['POST'])
def add_person():
    name = request.form.get('name','').strip()
    file = request.files.get('image')
    if not name or not file or not allowed_file(file.filename, ALLOWED_IMAGE):
        flash('Provide a name and a valid image.')
        return redirect(url_for('index'))
    person_dir = os.path.join(KNOWN_FOLDER, secure_filename(name))
    os.makedirs(person_dir, exist_ok=True)
    filename = secure_filename(file.filename)
    save_path = os.path.join(person_dir, filename)
    file.save(save_path)
    if os.path.exists(ENCODE_CACHE):
        os.remove(ENCODE_CACHE)
    flash(f'Added {name}')
    return redirect(url_for('index'))

@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    file = request.files.get('image')
    if not file or not allowed_file(file.filename, ALLOWED_IMAGE):
        flash('Upload a valid image.')
        return redirect(url_for('index'))

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    known_encs, known_names, rep_images = load_known_faces()

    img_bgr = cv2.imread(path)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model='hog')
    encs = face_recognition.face_encodings(rgb, boxes)

    labels = []
    match_details = []
    for enc in encs:
        distances = face_recognition.face_distance(known_encs, enc) if known_encs else np.array([])
        label = 'Unknown'
        top_matches = []
        if len(distances):
            idx_sorted = np.argsort(distances)[:5]
            for idx in idx_sorted:
                if distances[idx] <= 0.6:
                    person = known_names[idx]
                    top_matches.append({'name': person, 'distance': float(distances[idx]), 'rep_image': rep_images.get(person)})
            if top_matches:
                label = top_matches[0]['name']
        labels.append(label)
        match_details.append(top_matches)

    annotated = annotate_image(img_bgr.copy(), [(t,r,b,l) for t,r,b,l in boxes], labels)
    out_name = f"processed_{filename}"
    out_path = os.path.join(UPLOAD_FOLDER, out_name)
    cv2.imwrite(out_path, annotated)

    detected = labels if labels else ['No faces found']
    add_history({'source': filename, 'detected': detected})

    matched_refs = []
    for top_list in match_details:
        for m in top_list:
            if m['rep_image'] and m['rep_image'] not in matched_refs:
                matched_refs.append(m['rep_image'])

    return render_template('result.html',
                           original=url_for('uploaded_file', filename=filename),
                           processed=url_for('uploaded_file', filename=out_name),
                           detected=detected,
                           matched_refs=[url_for('static_file', path=p.replace('static'+os.sep,'')) if p.startswith('static'+os.sep) else url_for('known_file', filename=p.replace('known'+os.sep, '')) for p in matched_refs],
                           snapshots=[],
                           title='Recognition Result')

@app.route('/recognize_video', methods=['POST'])
def recognize_video():
    file = request.files.get('video')
    if not file or not allowed_file(file.filename, ALLOWED_VIDEO):
        flash('Upload a valid video.')
        return redirect(url_for('index'))

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    known_encs, known_names, rep_images = load_known_faces()

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(int(fps // 5), 1)
    snapshots = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        encs = face_recognition.face_encodings(rgb, boxes)

        for (top, right, bottom, left), enc in zip(boxes, encs):
            distances = face_recognition.face_distance(known_encs, enc) if known_encs else np.array([])
            label = 'Unknown'
            if len(distances):
                idx = int(np.argmin(distances))
                if distances[idx] <= 0.6:
                    label = known_names[idx]
            if label != 'Unknown':
                seconds = frame_idx / fps
                mm = int(seconds // 60)
                ss = int(seconds % 60)
                ts = f"{mm:02d}:{ss:02d}"
                crop = frame[max(0, top-10):bottom+10, max(0, left-10):right+10]
                if crop.size == 0:
                    crop = frame
                snap_name = f"{uuid.uuid4().hex}_{label}_{ts.replace(':','-')}.jpg"
                snap_path = os.path.join(SNAPSHOT_FOLDER, snap_name)
                cv2.putText(crop, f"{label} @ {ts}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imwrite(snap_path, crop)
                snapshots.append({'label': label, 'timestamp': ts, 'url': url_for('static', filename=f'snapshots/{snap_name}')})
        frame_idx += 1
    cap.release()

    detected_summary = [f"{s['label']} @ {s['timestamp']}" for s in snapshots] or ['No recognized faces']
    add_history({'source': filename, 'detected': detected_summary})

    return render_template('result.html',
                           original=None,
                           processed=None,
                           detected=detected_summary,
                           matched_refs=[],
                           snapshots=snapshots,
                           title='Video Recognition Snapshots')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/<path:path>')
def static_file(path):
    return send_from_directory(os.path.join(APP_ROOT, 'static'), path)

@app.route('/known/<path:filename>')
def known_file(filename):
    return send_from_directory(KNOWN_FOLDER, filename)

# -------- Webcam live recognition --------

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/webcam_feed')
def webcam_feed():
    import time

    def open_first_camera(max_index=5):
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"[INFO] Using camera index {i}")
                return cap
        raise RuntimeError("No available camera found")

    def gen():
        cap = open_first_camera()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        known_encs, known_names, rep_images = load_known_faces()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model='hog')
            encs = face_recognition.face_encodings(rgb, boxes)

            labels = []
            for enc in encs:
                distances = face_recognition.face_distance(known_encs, enc) if known_encs else np.array([])
                label = "Unknown"
                if len(distances):
                    idx = int(np.argmin(distances))
                    if distances[idx] <= 0.6:
                        label = known_names[idx]
                labels.append(label)

            frame = annotate_image(frame, boxes, labels)
            ok, jpeg = cv2.imencode('.jpg', frame)
            if not ok:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)

        cap.release()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----- Session permanence -----
@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(days=3)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)

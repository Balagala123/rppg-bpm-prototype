import cv2
import os
import numpy as np
import torch
from Model.DeepPhys import DeepPhys

# -----------------------
# Setup
# -----------------------
video_path = "/Users/sowmyabalagala/Downloads/wiseAI_Assesment/Person_Talking.mp4"

model = DeepPhys()
model.eval()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------
# Face detection
# -----------------------
def face_detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    return faces[0] if len(faces) > 0 else None

# -----------------------
# ROI extraction
# -----------------------
def get_roi(frame, face):
    x, y, w, h = face
    pad = 0.2

    x1 = max(int(x - w * pad), 0)
    y1 = max(int(y - h * pad), 0)
    x2 = min(int(x + w * (1 + pad)), frame.shape[1])
    y2 = min(int(y + h * (1 + pad)), frame.shape[0])

    return frame[y1:y2, x1:x2]

# -----------------------
# BPM from signal
# -----------------------
def bpm(signal, fps):
    signal = np.array(signal)
    signal = signal - np.mean(signal)

    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)

    mask = (freqs >= 0.7) & (freqs <= 3.0)

    fft = np.abs(fft[mask])
    freqs = freqs[mask]

    if len(freqs) == 0:
        return None

    peak = freqs[np.argmax(fft)]
    return peak * 60

# -----------------------
# Run DeepPhys on chunk
# -----------------------
def run_chunk(frames):
    processed = []

    for f in frames:
        f = cv2.resize(f, (36, 36))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = f / 255.0
        f = f.transpose(2, 0, 1)
        processed.append(f)

    processed = np.array(processed)

    motion = processed[1:] - processed[:-1]
    appearance = processed[1:]

    inp = np.concatenate([motion, appearance], axis=1)
    inp = torch.tensor(inp, dtype=torch.float32)

    signal = []

    with torch.no_grad():
        for i in range(len(inp)):
            out = model(inp[i].unsqueeze(0))
            signal.append(out.item())

    return signal

# -----------------------
# MAIN PIPELINE (CHUNK FIRST)
# -----------------------
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = fps if fps > 1 else 30

chunk_size = fps * 5

model_results = []
frames = []

# Face detection ON FIRST FRAME (for simplicity)
ret, first_frame = cap.read()
if not ret:
    raise Exception("Video not loaded")

face = face_detect(first_frame)
if face is None:
    raise Exception("No face detected")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# -----------------------
# CHUNK-FIRST LOOP
# -----------------------
while True:

    frames = []

    # Build one 5-sec chunk
    for i in range(chunk_size):

        ret, frame = cap.read()
        if not ret:
            break

        roi = get_roi(frame, face)
        if roi is not None:
            frames.append(roi)

    if len(frames) == 0:
        break

    # Run model on chunk
    signal = run_chunk(frames)

    bpm_value = bpm(signal, fps)
    model_results.append(bpm_value)

    print("Chunk BPM:", bpm_value)

cap.release()

# -----------------------
# FINAL OUTPUT
# -----------------------
valid = [b for b in model_results if b is not None]

if len(valid) > 0:
    print("Overall BPM:", np.mean(valid))
else:
    print("No valid BPM")
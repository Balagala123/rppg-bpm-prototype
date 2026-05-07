import cv2
import os
import numpy as np
import torch
import time
from Model.DeepPhys import DeepPhys
from mtcnn import MTCNN

video_path = "/Users/sowmyabalagala/Downloads/wiseAI_Assesment/Person_Talking.mp4"
base_dir = "/Users/sowmyabalagala/Downloads/wiseAI_Assesment"
output_folder = os.path.join(base_dir,"chunks")
os.makedirs(output_folder,exist_ok=True)
model = DeepPhys()
model.eval()

#face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
face_detector = MTCNN()
def face_detect(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = face_detector.detect_faces(rgb)
    if len(detections) == 0:
        return None
    x, y, w, h = detections[0]['box']
    x, y = max(0, x), max(0, y)
    return (x, y, w, h)

def get_roi(frame,face):
    x,y,w,h = face
    padding = 0.2

    x1 = max(int(x-w*padding),0)
    y1 = max(int(y - h * padding), 0)
    x2 = min(int(x + w * (1 + padding)), frame.shape[1])
    y2 = min(int(y + h * (1 + padding)), frame.shape[0])

    return frame[y1:y2,x1:x2]

def bpm(signal,frames_per_second):
    signal = np.asarray(signal)
    signal = signal-np.mean(signal)
    fft = np.fft.rfft(signal)
    heart_rate_frequencies = np.fft.rfftfreq(len(signal),d=1.0/frames_per_second)
    bpm_mask = (heart_rate_frequencies>=1.0) & (heart_rate_frequencies<=2.5)
    fft = np.abs(fft[bpm_mask])
    heart_rate_frequencies = heart_rate_frequencies[bpm_mask]
    if len(heart_rate_frequencies) ==0:
        return None
    peak_freq = heart_rate_frequencies[np.argmax(fft)]
    bpm = peak_freq*60
    return bpm

def smooth(values, smoothing_factor=0.6):
    smoothed_bpm = []
    for bpm_estimate in values:
        if bpm_estimate is None:
            continue
        if len(smoothed_bpm) == 0:
            smoothed_bpm.append(bpm_estimate)
        else:
            smoothed_bpm.append(smoothing_factor * smoothed_bpm[-1] + (1 - smoothing_factor )* bpm_estimate)
    return smoothed_bpm

def run_chunk(frames,frames_per_second):
    if len(frames)<2:
        return None
    processed_frames = []
    for f in frames:
        f=cv2.resize(f,(36,36))
        f=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
        f = f/255.0
        f = f.transpose(2,0,1)
        processed_frames.append(f)
    processed_frames = np.array(processed_frames)
    motion = processed_frames[1:]-processed_frames[:-1]
    appearance = processed_frames[1:]
    input_tensor = np.concatenate([motion,appearance],axis=1)
    input_tensor = torch.tensor(input_tensor,dtype=torch.float32)
    signal = []
    with torch.no_grad():
        signal = model(input_tensor)
    return bpm(signal.squeeze(), frames_per_second)

video_capture = cv2.VideoCapture(video_path)
frames_per_second = int(video_capture.get(cv2.CAP_PROP_FPS))
print(frames_per_second)
if frames_per_second is None or frames_per_second <= 1:
    frames_per_second = 30
frames_per_second = int(frames_per_second)
#30 frames->1 sec, 5sec->30*5
chunk_size = frames_per_second*5
print(chunk_size)
frame_read, first_frame = video_capture.read()
face = face_detect(first_frame)
if face is None:
    raise Exception("No face detected")
video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_index = 0
chunk_index = 0
video_writer = None
frames=[]
model_results= []

start_time = time.time()
while True:
    frame_read,current_frame = video_capture.read()
    if not frame_read:
        break
    roi = get_roi (current_frame,face)
    if roi is None:
        continue
    frames.append(roi)
    if len(frames) == chunk_size:
        height,width = frames[0].shape[:2]
        chunk_path = os.path.join(output_folder, f"chunk_{chunk_index}.mp4")
        video_writer = cv2.VideoWriter(chunk_path,cv2.VideoWriter_fourcc(*'mp4v'),frames_per_second,(width, height),True)
        for f in frames:
            video_writer.write(f.astype(np.uint8))

        video_writer.release()
        final_bpm = run_chunk(frames,frames_per_second)
        model_results.append(final_bpm)
        print(f"Chunk {chunk_index} BPM: {final_bpm}")
        frames = []
        chunk_index += 1
video_capture.release()
end_time = time.time()

valid_bpm = [bpm_value for bpm_value in model_results if bpm_value is not None]

if len(valid_bpm)>0:
    overall_bpm = np.mean(valid_bpm)
    print(f"Overall_bpm:{overall_bpm}")
else:
    print("No valid BPM values found")
total_latency = end_time-start_time
num_chunks_processed = len(model_results)
processing_speed = num_chunks_processed/total_latency if total_latency>0 else 0
print(f"Latency(Inference Time):{total_latency:.2f} seconds")
print(f"Processing Speed: {processing_speed:.2f}")
#filtered_bpm = [bpm_value for bpm_value in model_results if bpm_value is not None]
smoothed_bpm = smooth(valid_bpm)
if len(smoothed_bpm)>0:
    mean_bpm = np.mean(smoothed_bpm)
    std_bpm = np.std(smoothed_bpm)
    min_bpm = np.min(smoothed_bpm)
    max_bpm = np.max(smoothed_bpm)
    print(f"Mean BPM (Final 60s Estimate): {mean_bpm:.2f}")
    print(f"BPM Stability (Std Dev): {std_bpm:.2f}")
    print(f"Min BPM: {min_bpm:.2f}")
    print(f"Max BPM: {max_bpm:.2f}")
else:
    print("No BPM values are found")





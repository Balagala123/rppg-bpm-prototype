import cv2
import os
from Model.DeepPhys import DeepPhys
import numpy as np
import torch
import time
video_path = "/Users/sowmyabalagala/Downloads/wiseAI_Assesment/Person_Talking.mp4"
output_folder = "chunks"
os.makedirs(output_folder,exist_ok=True)
model = DeepPhys()
model.eval()
video_capture = cv2.VideoCapture(video_path)
frames_per_second = int(video_capture.get(cv2.CAP_PROP_FPS))
print(frames_per_second)
if frames_per_second is None or frames_per_second <= 1:
    frames_per_second = 30
frames_per_second = int(frames_per_second)
#30 frames->1 sec, 5sec->30*5
chunk_size = frames_per_second*5
print(chunk_size)
frame_index = 0
chunk_index = 0
video_writer = None
frames=[]
while True:
    frame_read,current_frame = video_capture.read()
    if not frame_read:
        break
    if len(current_frame.shape) == 2:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)

    elif current_frame.shape[2] == 1:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
    frames.append(current_frame)
video_capture.release()
for i in range(0,len(frames),chunk_size):
    chunk_frames = frames[i:i+chunk_size]
    if len(chunk_frames)==0:
        continue
    height,width,num_channels = chunk_frames[0].shape
    if len(chunk_frames)<frames_per_second:
        continue
    output_video_path = f"chunks/chunk{chunk_index}.mp4"
        #height,width,num_channels = current_frame.shape
    video_writer = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*'mp4v'),frames_per_second,(width,height),True)
    for frame in chunk_frames:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame)
    video_writer.release()
    chunk_index+=1

def bpm(signal,frames_per_second):
    signal = np.asarray(signal)
    signal = signal-np.mean(signal)
    fft = np.fft.rfft(signal)
    heart_rate_frequencies = np.fft.rfftfreq(len(signal),d=1.0/frames_per_second)
    bpm_mask = (heart_rate_frequencies>=1.0) & (heart_rate_frequencies<=2.0)
    fft = np.abs(fft[bpm_mask])
    heart_rate_frequencies = heart_rate_frequencies[bpm_mask]
    peak_freq = heart_rate_frequencies[np.argmax(fft)]
    bpm = peak_freq*60
    return bpm

def run_chunk(chunk_path):
    video_capture= cv2.VideoCapture(chunk_path)
    frames = []
    while True:
        frame_read,frame = video_capture.read()
        if not frame_read:
            break
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.resize(frame,(36,36))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = frame.transpose(2,0,1)
        frames.append(frame)
    video_capture.release()
    frames = np.array(frames)
    #Filter invalid chunks
    if len(frames)<2:
        return None
    motion = frames[1:]-frames[:-1]
    appearance = frames[1:]
    input_tensor = np.concatenate([motion,appearance],axis=1)
    input_tensor = torch.tensor(input_tensor,dtype=torch.float32)
    #print("Input tensor shape:", input_tensor.shape)
    with torch.no_grad():
            output = model(input_tensor)
    return bpm(output.squeeze(), frames_per_second)

start_time = time.time()
model_results = []
for file in sorted(os.listdir(output_folder)):
    if not file.endswith(".mp4"):
        continue
    chunk_path = os.path.join(output_folder,file)
    final_bpm = run_chunk(chunk_path)
    model_results.append(final_bpm)
    print(f"{file} -> BPM:{final_bpm}")
end_time = time.time()
#print(model_results[:2])
valid_bpm = [bpm_value for bpm_value in model_results if bpm_value is not None and 60<=bpm_value<=120]
if len(valid_bpm)>0:
    overall_bpm = np.mean(valid_bpm)
    print(f"Overall_bpm:{overall_bpm}")
else:
    print("No valid BPM values found")
total_latency = end_time-start_time
num_chunks_processed = len(model_results)
inference_fps = num_chunks_processed/total_latency if total_latency>0 else 0
print(f"Latency(Inference Time):{total_latency:.2f} seconds")
print(f"Frames Per Second (FPS): {inference_fps:.2f}")
valid_bpm = [bpm_value for bpm_value in model_results if bpm_value is not None]
if len(valid_bpm)>0:
    mean_bpm = np.mean(valid_bpm)
    std_bpm = np.std(valid_bpm)
    min_bpm = np.min(valid_bpm)
    max_bpm = np.max(valid_bpm)
    print(f"Mean BPM (Final 60s Estimate): {mean_bpm:.2f}")
    print(f"BPM Stability (Std Dev): {std_bpm:.2f}")
    print(f"Min BPM: {min_bpm:.2f}")
    print(f"Max BPM: {max_bpm:.2f}")
else:
    print("No BPM values are found")



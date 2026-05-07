**#MODEL PERFORMANCE**
I have chosen the DeepPhys-based pipeline, which produced relatively stable BPM estimates across most 5-second chunks, with values remaining within a realistic heart rate range throughout the 60-second video. Face detection methods such as Haar Cascade and MTCNN, along with ROI extraction, helped reduce background noise and improve signal quality before passing the frames to the model. Some variation between chunks was observed due to facial motion, which are common challenges in rPPG systems. Overall, the model maintained stable BPM estimation performance while achieving near real-time processing speed.

#**Latency**
The total latency for processing my test 60-second video is 7.60 seconds. The system runs faster than the actual video duration, making it efficient and suitable for real-time processing.

#**Failure cases**
The pipeline may produce less reliable BPM values when there is significant motion blur or head movement, as this affects the quality of facial signal extraction. Performance also degrades when the face is partially covered or not clearly visible in the video. Noise in BPM estimates due to motion and signal variations can occur across chunks, but this is reduced by keeping valid BPM values within a normal heart range. Another issue I observed was inconsistent chunk generation, where an extra or incomplement segemnet was produced during video splitting. I handled this by filtering invalid chunks to ensure correct 5-second segmentation before processing. Very low lighting or poor video quality can also reduce the accuracy of rPPG signal extraction.

#**AI Tools**
I have used ChatGPT to understand the rPPG concept and DeepPhys model behavior. I have also used it for refining the README documentation for better presentation.


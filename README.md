**#MODEL PERFORMANCE**
I have chosen the DeepPhys model from the provided rPPg toolbox and used it as the core model in the 5-second chunk-based pipeline. In my test video, the model produced consistent BPM estimates across most chunks, with values staying within a normal heart rate range. Some variation was observed between chunks due to motion and signal noise, but overall the estimates are stable throughout the 60-second video. The input frames are processed before being passed to the model to ensure consistent input. Averaging the chunk-level outputs helped reduce fluctuations and resulted in a  stable final BPM estimate for the video.

#**Latency**
The total latency for processing my test 60-second video is 7.60 seconds. The system runs faster than the actual video duration, making it efficient and suitable for real-time processing.

#**Failure cases**
The pipeline may produce less reliable BPM values when there is significant motion blur or head movement, as this affects the quality of facial signal extraction. Performance also degrades when the face is partially covered or not clearly visible in the video. Noise in BPM estimates due to motion and signal variations can occur across chunks, but this is reduced by keeping valid BPM values within a normal heart range. Another issue I observed was inconsistent chunk generation, where an extra or incomplement segemnet was produced during video splitting. I handled this by filtering invalid chunks to ensure correct 5-second segmentation before processing. Very low lighting or poor video quality can also reduce the accuracy of rPPG signal extraction.

#**AI Tools**
I have used ChatGPT to understand the rPPG concept and DeepPhys model behavior. I have also used it for refining the README documentation for better presentation.


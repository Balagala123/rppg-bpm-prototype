**#MODEL PERFORMANCE**

I have chosen the DeepPhys model from the provided rPPg toolbox and used it as the core model in the 5-second chunk-based pipeline. In my test video, the model produced consistent BPM estimates across most chunks, with values staying within a normal heart rate range. Some variation was observed between chunks due to motion and signal noise, but overall the estimates are stable throughout the 60-second video. The input frames are processed before being passed to the model to ensure consistent input. Averaging the chunk-level outputs helped reduce fluctuations and resulted in a  stable final BPM estimate for the video.



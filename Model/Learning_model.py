#x_shpe in attention mask->(b->batch size,c->channels, h,w)
#batch size is the number of training samples the model processes before it updates its weights once.
#1. Add everything in the image (per channel)
# → finds total “energy”
# 2. Divide each pixel by total
# → finds importance of each pixel
# 3. Multiply back by image size
# → restores scale
# 4. Multiply by 0.5
# → dampens output
#This is a method used for saving and recreating a model layer later.(get_config)
#Part	Meaning, first 3 channels	motion (difference frames), last 3 channels	appearance (raw frames), diff_input = inputs[:, :3, :, :],raw_input = inputs[:, 3:, :, :]
#Appearance -> but for raw image learns skin color + lighting + static features
#Motion -> how pixels are changing over time
#Pooling = shrinking the image / feature map
# It reduces:
# height (H)
# width (W)
# But keeps:
# important information
#dropout:“Randomly turn off neurons during training”, prevents overfitting(In neural networks, overfitting is a condition where the model learns the training data too closely, including its noise and random fluctuations, so it performs well on training data but poorly on new, unseen data.),forces robustness,improves generalization
#Input image → CNN layers → pooling → feature maps → flatten → Linear layer
#After CNN + pooling, the image becomes a flattened vector, and its size depends on input image size.
#automatically learns important features (like edges, shapes, textures) from images using filters.
#input->one output layer->fully connceted layer
#Gating means controlling information flow using another signal, element-wise multiplication used to filter information, filters features using attention
#tanh->keeps signal centered around 0, good for feature learning
#HCW->CHW ->transpose, array(frames)->stack frames->(N,3,36,36),N->number of frames
#You duplicate channels:(3 channels → 6 channels), So now:(N, 6, 36, 36) axis=1->channels
#unsqueeze add batch dimensions
#motion = current_frame - previous_frame


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from tqdm import tqdm
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array, vfx, concatenate_videoclips

batch_id = int(sys.argv[1])
print(batch_id)

folders = ['logs/1230_test/videos/', 'logs/1231_test/videos/', 'logs/1232_test/videos/', 'logs/1233_test/videos/']
final   = 'logs/TEST_2/'

try:
    os.mkdir(final)
except:
    pass

files        = np.sort([i for i in os.listdir(folders[0]) if ".mp4" in i])
batch_length = len(files)//100
start_       = batch_id*(batch_length+1)
end_         = (batch_id+1)*(batch_length+1)
if(start_>len(files)): exit()
if(end_  >len(files)): end_ = len(files)
files    = files[start_:end_] if batch_id>=0 else files
    
    
for n, file in enumerate(files):
    print(n, len(files), file) 
    clips = []
    for i, folder in enumerate(folders):
        if(os.path.isfile(folder+file)):
            clip = VideoFileClip(folder+file)
            T    = clip.duration
            # clips.append(clip.subclip(T/4*i, T/4*(i+1)))
            clips.append(clip)
    
    try:
        if(len(clips)==len(folders)):
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(final + file, fps=30, codec='libx264', audio_codec='aac')
    except:
        pass

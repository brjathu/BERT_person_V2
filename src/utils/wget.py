import os
import wget
import numpy as np
from tqdm import tqdm

save_path = "/private/home/jathushan/videos/webm/"
vvt_files = np.loadtxt("/private/home/jathushan/videos/list.txt", dtype=str)
for file in tqdm(vvt_files):
    if(os.path.exists(os.path.join(save_pathsave_path, file))):
        continue
    else:
        wget.download("https://ir.nist.gov/tv_vtt_data/Video_Files/"+file, save_path)
        print(file)
        # break
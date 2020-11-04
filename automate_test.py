
import os, glob
import random
from tqdm import tqdm

file_list = [f for i,f in enumerate(glob.glob("/pless_nfs/home/datasets/Hotels-50K/images/test/unoccluded/**/*.jpg", recursive=True)) if "SEG" not in f]
random.shuffle(file_list)

for img in tqdm(file_list):
    bashCommand = "python -u test.py --imgs {} --gpu 0 --cfg config/current_experiments/162-class_Hotels_and_ADE.yaml "
    bashCommand = bashCommand.format(img)
    # print(bashCommand)
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


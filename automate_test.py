
import os, glob
import random
from tqdm import tqdm
from multiprocessing import Process

def worker(file_list, gpu):
    for img in file_list:
        bashCommand = "python -u test.py --imgs {} --gpu {} --cfg /pless_nfs/home/mdt_/semantic-segmentation-pytorch/config/current_experiments/exp15-remake.yaml "
        bashCommand = bashCommand.format(img, gpu)
        print(bashCommand)
        import subprocess
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
    

file_list = [f for i,f in enumerate(glob.glob("/pless_nfs/home/datasets/Hotels-50K/images/train/92/99785/traffickcam/**/*.jpg", recursive=True)) if "SEG" not in f]

worker(file_list, 1)
# p1 = Process(target=worker, args=(file_list[:len(file_list) // 2], 3, ))
# p2 = Process(target=worker, args=(file_list[len(file_list) // 2:], 4, ))

# p1.start()
# p2.start()

# p1.join()
# p2.join()

import os, glob
import random
from tqdm import tqdm
from multiprocessing import Process

def worker(file_list, gpu):
    print("In process: ", gpu+1)
    for i, img in enumerate(file_list):
        bashCommand = "python -u test.py --imgs {} --gpu {} --cfg /pless_nfs/home/mdt_/semantic-segmentation-pytorch/config/current_experiments/exp15-remake.yaml "
        bashCommand = bashCommand.format(img, gpu)
        # print(bashCommand)
        import subprocess
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("Process {}:\n\tPercent Complete: {}".format(gpu+1, i /len(file_list)))
    print("Fisnished process: ", gpu+1)
    

file_list = [f for i,f in enumerate(glob.glob("/pless_nfs/home/datasets/Hotels-50K/images/test/unoccluded/**/*.jpg", recursive=True)) if "SEG" not in f]

p1 = Process(target=worker, args=(file_list[:len(file_list) // 2], 0, ))
p2 = Process(target=worker, args=(file_list[len(file_list) // 2:], 1, ))

p1.start();p2.start()

p1.join();p2.join()
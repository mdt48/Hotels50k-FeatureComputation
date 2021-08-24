
from collections import namedtuple
from multiprocessing import pool
from pathlib import Path
from types import prepare_class
import pandas as pd
from tqdm import tqdm
import random
import torch.nn as nn
import torch
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from scipy.spatial import distance
import json
import time

import multiprocessing as mp


random.seed(42)
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
root = ""
c = 0
ade = pd.read_csv("data/features_150.csv")
ade_classes = {row["Idx"]:row["Name"].replace(";", "-") for idx, row in ade.iterrows()}

def calc_distribution(same, diff):
    kde_same = KernelDensity(kernel="gaussian",bandwidth=0.75).fit(np.array(same).reshape(-1, 1))
    kde_diff = KernelDensity(kernel="gaussian",bandwidth=0.75).fit(np.array(diff).reshape(-1, 1))
    return kde_same, kde_diff

def make_plot(path, same, diff,title, feature=None):
    Path(path).mkdir(parents=True, exist_ok=True)
    

    
    fig, ax = plt.subplots()
    names = ["Same", "Diff"]
    for idx, a in enumerate([same, diff]):
        sns.distplot(a, ax=ax, kde=True, hist=False, rug=False, label=names[idx])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 12.5])
    

    fig.add_axes(ax)
    plt.legend()
    fig.suptitle(title, fontsize=10)
    plt.savefig(path+"same-diff-dist.png")
    
    plt.close('all')

def load_pckl(path):
    with open(path, "rb") as f: ret = pickle.load(f)

    return ret

def closest(current_image, current_image_index, images, current_feature, diff):
    closest = 0

    # current_feature = load_pckl(current_image + '/indiv_features.pckl')[current_image_index]
    

    for idx, img in images.iterrows():
        if current_image == img["Path"]: 
            continue
        # diff_feature = load_pckl(img["Path"] + '/indiv_features.pckl')[img["Idx"]]
        try:
            # diff_feature = load_pckl(img["Path"] + '/indiv_features.pckl')[img["Idx"]]
            diff_feature = torch.load(img["Path"] + '/indiv_features.pt')[img["Idx"]]
        except:
            continue

        dist =  float(cos(current_feature, diff_feature))
        if dist == 1.0: 
            return dist;
        closest = max(closest,  dist)
 
    return closest

def worker(csv, cl, mode):
    same_class = csv.loc[csv["Class"] == cl].drop_duplicates()
    diff_class = csv.loc[csv["Class"] != cl].drop_duplicates()

    same_class.reset_index(drop=True, inplace=True)
    if len(same_class) == 0: return

    diff_class.reset_index(drop=True, inplace=True)

    closest_same_class = []
    closest_diff_class = []

    sample_length = min( 1000, len(same_class) )

    
    for idx, d in tqdm(same_class.sample(sample_length).iterrows(), total=sample_length, desc="Closest"):
    # for idx, d in same_class.sample(sample_length).iterrows():
        try: current_feature =   torch.load(d["Path"] + '/indiv_features.pt')[d["Idx"]]
        except: continue

        
        best_same = closest(d["Path"], d["Idx"],same_class.sample(sample_length), current_feature, False)
        best_diff = closest(d["Path"], d["Idx"], diff_class.sample(int(sample_length*1.5)), current_feature, True)

        if not best_same or not best_diff:
            continue

        closest_same_class.append( best_same )
        closest_diff_class.append( best_diff )

    if not closest_same_class or not closest_diff_class: return
    path = "data/figures/{}/closest_same_closest_diff/{}/".format(mode,ade_classes[cl])
    title = "{} - Same-Diff Class".format(ade_classes[cl])
    make_plot(path, closest_same_class, closest_diff_class, title)

    kde_same, kde_diff = calc_distribution(closest_same_class, closest_diff_class)

    with open(path + "distribution.pckl", "wb") as f:
        pickle.dump([kde_same, kde_diff], f)
        
def main(mode):
    csv = pd.read_csv("data/{}/features.csv".format(mode), names=["Idx", "Class", "Path"])
    csv["Instance"] = [row.split("/")[3] for row in csv["Path"]]
    # instances = csv["Instance"].unique()
    classes = csv["Class"].unique()

    for cl in tqdm(classes[:len(classes) // 2], desc="Total"):
        worker(csv, cl, mode)
if __name__ == "__main__":
    main("train_non_torch")
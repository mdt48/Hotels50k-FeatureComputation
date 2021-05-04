import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity, NearestNeighbors
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
random.seed(42)

rng = np.random.default_rng()
cos = nn.CosineSimilarity(dim=0, eps=1e-6)

ade = pd.read_csv("data/features_150.csv")
ade_classes = {row["Idx"]:row["Name"].replace(";", "-") for idx, row in ade.iterrows()}

def get_classes(csv):
    unfilter_classes = list(csv["Class"].unique())[1:]
    classes = set()
    for cl in unfilter_classes:
        if "data" in cl: continue
        classes.add(cl)
    return list(classes)

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

    
def my_distance(x, y, **kwargs):
    # print(kwargs)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    return 1- float( cos(torch.tensor(x), torch.tensor(y)) )

def analysis(mode, cl, closest_same_class, closest_diff_class):
    path = "data/figures/{}/closest_same_closest_diff/{}/".format(mode+"-take-2",ade_classes[int(cl)])
    title = "{} - Same-Diff Class".format(ade_classes[int(cl)])
    make_plot(path, closest_same_class, closest_diff_class, title)

    kde_same, kde_diff = calc_distribution(closest_same_class, closest_diff_class)

    with open(path + "distribution.pckl", "wb") as f:
        pickle.dump([kde_same, kde_diff], f)

def closest(csv, cl, mode, same, diff):
    if len(same) == 0: return
    closest_same_class = []
    closest_diff_class = []

    sample_length = min( 750, len(same) )

    same_class = rng.choice(same, sample_length, axis=0)
    # same_class = same
    
    
    for idx, feature in tqdm(enumerate(same_class), total=len(same_class), desc="Closest"):
        
        search_index = np.delete(same, idx, 0)
        if len(search_index) == 0: return
        knn = NearestNeighbors(algorithm="brute", n_neighbors=1, metric=my_distance)
        knn.fit(search_index)
        closest_same_class.append( 1-knn.kneighbors(feature.reshape(1, -1))[0][0][0] )


        
        closest_diff_class.append( 1 - diff.kneighbors(feature.reshape(1, -1))[0][0][0] )




    if  closest_same_class and  closest_diff_class :
        analysis(mode, cl, closest_same_class, closest_diff_class)
    else:
        return


        
def main(mode):
    csv = pd.read_csv("data/{}/features.csv".format(mode), names=["Idx", "Class", "Path"], low_memory=False)
    classes = get_classes(csv)
    knn_diff = NearestNeighbors(algorithm="brute", n_neighbors=1, metric=my_distance)


    # for cl in tqdm(classes[:len(classes) // 4], desc="Total"):
    # for cl in tqdm(classes[len(classes) // 4:len(classes) // 2], desc="Total"):
    # for cl in tqdm(classes[len(classes) // 2:(len(classes) // 2)+len(classes) // 4], desc="Total"):
    # for cl in tqdm(classes[(len(classes) // 2)+len(classes) // 4:], desc="Total"):
    for cl in ['8']:
        try: same = load_pckl('data/{}/same_index.pckl'.format(mode))[cl]
        except: continue
        knn_diff.fit(rng.choice(load_pckl('data/{}/diff_index.pckl'.format(mode))[cl], 20000, axis=0))
        closest(csv, cl, mode, same, knn_diff)
if __name__ == "__main__":
    main("train_non_torch")

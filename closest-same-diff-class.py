
from pathlib import Path
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
from sklearn.neighbors import NearestNeighbors



random.seed(42)
rng = np.random.default_rng()
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

    
def my_distance(x, y, **kwargs):
    # print(kwargs)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    return float( cos(torch.tensor(x), torch.tensor(y)) )

def worker(csv, cl, mode, same, diff):
    # same_class = same[cl]
    # diff_class = diff[cl]

    closest_same_class = []
    closest_diff_class = []

    sample_length = min( 2000, len(same) )

    same_class = rng.choice(same, sample_length, axis=0)
    
    # for idx, d in tqdm(same_class.sample(sample_length).iterrows(), total=sample_length, desc="Closest"):
    for idx, feature in tqdm(enumerate(same_class), total=len(same_class)):

        search_index = np.delete(same_class, idx, 0)
        knn = NearestNeighbors(algorithm="brute", n_neighbors=1, metric=my_distance)
        knn.fit(search_index)
        closest_same_class.append( knn.kneighbors(feature.reshape(1, -1))[0][0][0] )


        knn = NearestNeighbors(algorithm="brute", n_neighbors=1, metric=my_distance)
        knn.fit(diff)
        closest_diff_class.append( knn.kneighbors(feature.reshape(1, -1))[0][0][0] )




    if not closest_same_class or not closest_diff_class: return
    path = "data/figures/{}/closest_same_closest_diff/{}/".format(mode+"-take-3",ade_classes[cl])
    title = "{} - Same-Diff Class".format(ade_classes[cl])
    make_plot(path, closest_same_class, closest_diff_class, title)

    kde_same, kde_diff = calc_distribution(closest_same_class, closest_diff_class)

    with open(path + "distribution.pckl", "wb") as f:
        pickle.dump([kde_same, kde_diff], f)
        
def main(mode):
    csv = pd.read_csv("data/{}/features.csv".format(mode), names=["Idx", "Class", "Path"])
    csv["Instance"] = [row.split("/")[3] for row in csv["Path"]]
    csv["Chain"] = [row.split("/")[2] for row in csv["Path"]]

    
    
    classes = csv["Class"].unique()
    # same = load_pckl('data/test_non_torch/same_index.pckl')
    # diff = load_pckl('data/test_non_torch/diff_index.pckl')
    # instances = csv["Instance"].unique()
    # for cl in same.keys():
    #     same[cl] = same[cl].astype("float16")
    # for cl in diff.keys():
    #     diff[cl] = diff[cl].astype("float16")

    for cl in tqdm(classes[:len(classes) // 2], desc="Total"):
        same = load_pckl('data/test_non_torch/same_index.pckl')[cl]
        diff = load_pckl('data/test_non_torch/diff_index.pckl')[cl]
        worker(csv, cl, mode, same, diff)
if __name__ == "__main__":
    main("test_non_torch")
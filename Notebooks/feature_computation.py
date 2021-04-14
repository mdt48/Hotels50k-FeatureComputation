# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from tqdm import tqdm
import random
import itertools
import torch.nn as nn
import torch
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.figure_factory as ff
import os

random.seed(42)
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
root = "/pless_nfs/home/mdt_/Hotels50-FeatureComputation/"

# %%
def normalize_df(l1, l2):
    n = min(len(l1), 1000)

    l2 = l2.sample(n=n, random_state=42)
    l1 = l1.sample(n=n, random_state=42)
    return l1, l2

def normalize_lists(l1, l2):
    n = min(len(l1), 1000)
    l2 = random.sample(l2, n)
    l1 = random.sample(l1, n)
    return l1, l2

# %%
def sim(same, diff, cl):
    same = same.reset_index(drop=True)
    diff = diff.reset_index(drop=True)

    same_pairs = list(itertools.combinations(same.index.to_list(), 2))
    diff_pairs = list(itertools.combinations(diff.index.to_list(), 2))

    same_pairs, diff_pairs = normalize_lists(same_pairs, diff_pairs)
    

    root = "/pless_nfs/home/mdt_/Hotels50-FeatureComputation/"

    same_sim, diff_sim = [], []

    for item in zip(same_pairs, diff_pairs):
        s1 = root + same.iloc[item[0][0]]["Path"] + "/indiv_features1.pt"
        s2 = root + same.iloc[item[0][1]]["Path"] + "/indiv_features1.pt"
        
        d1 = root + diff.iloc[item[1][0]]["Path"] + "/indiv_features1.pt"
        d2 = root + diff.iloc[item[1][1]]["Path"] + "/indiv_features1.pt"

        same_sim.append( float(cos(torch.load(s1)[same.iloc[item[0][0]]["Idx"]], torch.load(s2)[same.iloc[item[0][1]]["Idx"]])) )
        diff_sim.append( float(cos(torch.load(d1)[diff.iloc[item[1][0]]["Idx"]], torch.load(d2)[diff.iloc[item[1][1]]["Idx"]])) )
        

    return same_sim, diff_sim


# %%
csv = pd.read_csv("/pless_nfs/home/mdt_/Hotels50-FeatureComputation/data/train/features.csv", names=["Idx", "Class", "Path"])
csv["Instance"] = [row.split("/")[3] for row in csv["Path"]]


# %%
instances = csv["Instance"].unique()
classes = csv["Class"].unique()


# %%
ade_classes = pd.read_csv("/pless_nfs/home/mdt_/Hotels50-FeatureComputation/data/features_150.csv")


# %%
for cl in tqdm(classes):
    total_same, total_diff = [], []

    instances = list(set(csv.loc[csv["Class"] == cl]["Instance"].to_list()))
    n = min(1000, len(instances))
    instances = random.sample(instances, n)

    for instance in tqdm(instances):
    
        same, diff = normalize_df(csv.loc[csv["Instance"] == instance], csv.loc[csv["Instance"] != instance])
        same_sim, diff_sim = sim(same, diff, cl)
        total_same.extend(same)
        total_diff.extend(diff)
        # print("SAME: " + str(same_sim), "DIFF: " + str(diff_sim), sep="\n")
        # kde_same = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.array(same_sim).reshape(-1, 1))
        # kde_diff = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.array(diff_sim).reshape(-1, 1))

        # x = np.array(same_sim).reshape(-1, 1)
        # x_d = np.linspace(-1, 2, 1000)
        # logprob = kde_same.score_samples(x_d[:, None])
        # plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
        # plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
        # plt.ylim(0, 2)


    group_labels = ['Same', 'Diff']
    colors = ['#333F44', '#37AA9C']

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot([total_same, total_diff], group_labels, show_hist=False, colors=colors, show_rug=False)

    # Add title
    fig.update_layout(title_text='Feature: {}- Similarities Same and Diff Inst of Hotels'.format(feature))
    fig.show()

    feature = ade_classes.loc[ade_classes["Idx"] == cl]["Name"]
    path = root+ "data/figures/train/{}/".format(feature)
    os.makedirs(path)

    fig.write_image(path + "same-diff-instances.png")

            


# %%
len(set(csv.loc[csv["Class"] == 6]["Instance"].to_list()))



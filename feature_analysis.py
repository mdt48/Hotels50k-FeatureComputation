# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os 
import pprint
import csv
from torch import chunk
from tqdm import tqdm
import pickle
import plotly.express as px
import plotly.graph_objects as go
from multiprocessing import Pool

# %%
def whole_image_avg(t):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    temp = torch.squeeze(t).permute(1,2,0).to(device)
    avg = torch.zeros((2048)).cpu().to(device)
    for i in range(temp.size()[0]):
        for j in range(temp.size()[1]):
            avg += temp[i][j]
    avg = torch.div(avg, temp.size()[0]* temp.size()[1])
    return avg


# %%
def imgs_for_hotel(p):
    same_hotel = {}
    for root, dirs, files in os.walk(os.path.join(p)):
        for d in dirs:
            for rs, ds, fs in os.walk(os.path.join(root, d)):
                t = []
                fs.sort(key=lambda f: os.path.splitext(f))
                for f in fs:
                    if "pt" in f or "csv" in f:
                        t.append(os.path.join(rs, f))
                # same_hotel[h][d] = t
                same_hotel[d] = t
    return same_hotel


# %%
def get_pairs(rooms):
    num_rooms = len(rooms.keys())
    hotel_id = [f for f in range(num_rooms)]
    book_dict = dict(zip(hotel_id, rooms))
    ids = list(rooms.keys())

    pairs = []
    for i, v in enumerate(ids):
        for j in ids[i+1:]:
            pairs.append((ids[i], j))
    return pairs


# %%
def hist_across_chain(data, k):
    vals = [data[key][0] for key in data.keys()]
    if len(vals) <= 3:
        return
    fig = go.Figure(data=[go.Histogram(x=vals)])
    path = os.path.join("hists/", k)
    if not os.path.exists(path):
        os.makedirs(path)
    fig.write_image(os.path.join(path,"similarity_same_hotel.png"))

def hist_per_room(data, k):
    for key in data.keys():
        vals = data[key][0]
        fig = go.Figure(data=[go.Histogram(x=[vals], histnorm='probability')])
        path = os.path.join("hists/", k, key)
        if not os.path.exists(path):
            os.makedirs(path)
        fig.write_image(os.path.join(path,"similarity_of_same_room.png"))

# %%
def cos_sim(pairs, rooms):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    pair_cos = {}
    for pair in pairs:
        h1 = torch.load(rooms[pair[0]][2]); h2 = torch.load(rooms[pair[1]][2])
        h1 = whole_image_avg(h1); h2 = whole_image_avg(h2)

        pair_cos[pair[0] + "-" + pair[1]] = float(cos(h1, h2)) 
    return pair_cos


def hists(data, k):
    hist_per_room(data, k)
    hist_across_chain(data, k)

def worker(chains):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Similarity = {}
    for ch in chains:
        print(ch)
        Similarity[ch] = {}
        for root, hotels, ___ in list(os.walk(os.path.join("features2", ch))):
            hotels = sorted(hotels, key=sorter)
            hotel_self_similarity = None
            hotel_vector = torch.zeros((2048)).to(device)
            for idx, h in enumerate(hotels):
                rooms = imgs_for_hotel(os.path.join(root, h))
                pairs = get_pairs(rooms)
                for r in rooms.keys():
                    if len(rooms[r]) == 3:
                        hotel_vector += whole_image_avg(torch.load(rooms[r][2]))
                if not pairs:
                    continue
                p_c = cos_sim(pairs, rooms)
                hotel_self_similarity = sum(p_c.values()) / len(p_c.values())
                Similarity[ch][h] = (hotel_self_similarity,hotel_vector / len(rooms.keys()))

        hists(Similarity[ch], ch)
        Similarity = {}
# %%
def sorter(x):
    return int(x)
                    

def main():
    for _, chains, __ in list(os.walk("features2")):
        chains = sorted(chains, key=sorter)
        worker(chains[1:])

# %%
if __name__ == "__main__":
    main()

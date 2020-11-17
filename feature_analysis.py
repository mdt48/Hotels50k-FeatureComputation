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
def calculate_hist_same_room(data, k):
    for key in data.keys():
        # vals = [f[0] for f in data[key].values()]
        vals = [data[key][0]]
    # p = plt.hist(vals, bins=15, histtype="stepfilled", alpha=0.3, color="r", density=True, label="Similarity Between Images of The Same Room")
        fig = go.Figure(data=[go.Histogram(x=vals)])
        path = os.path.join("hists/", k, key)
        if not os.path.exists(path):
            os.makedirs(path)
        fig.write_image(os.path.join(path,"similarity_same_hotel.png"))

def calulate_hist_per_room(data, k):
    for key in data.keys():
        vals = data[key][0]
        fig = go.Figure(data=[go.Histogram(x=[vals])])
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

def worker(chains):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Similarity = {}
    for ch in chains:
        print(ch)
        # print("Process: {}; Chain: {}".format(os.getpid(), ch))
        Similarity[ch] = {}
        for root, hotels, ___ in list(os.walk(os.path.join("features2", ch))):
            hotels = sorted(hotels, key=sorter)
            hotel_self_similarity = None
            hotel_vector = torch.zeros((2048)).to(device)
            for idx, h in enumerate(hotels):
                # print("Process: {}; Chain {}; Percentage Complete: {}".format(os.getpid(), ch, idx+1/len(hotels)))
                rooms = imgs_for_hotel(os.path.join(root, h))
                pairs = get_pairs(rooms)
                for r in rooms.keys():
                    if len(rooms[r]) == 3:
                        hotel_vector += whole_image_avg(torch.load(rooms[r][2]))
                if not pairs:
                    continue
                p_c = cos_sim(pairs, rooms)
                # print("Chain: {}; ID: {}; AVG Sim: {}".format(ch, h, hotel_self_similarity))
                hotel_self_similarity = sum(p_c.values()) / len(p_c.values())
                # hist = calculate_hist(p_c)
                Similarity[ch][h] = (hotel_self_similarity,hotel_vector / len(rooms.keys()))
            # for k in Similarity.keys():
                # print("Histogram for chain: " + ch)
                # if Similarity.values():
                #     # calulate_hist_per_room(Similarity[k], k)
            calculate_hist_same_room(Similarity[ch], ch)
            Similarity = {}
# %%
def sorter(x):
    return int(x)
def comp():
    hist = None

    # structure-->chain-->hotel-->similarity: (similarity for that hotel, the averaged vector for that hotel)
    # so similarity of the rooms in a certain hotel
    for _, chains, __ in list(os.walk("features2")):
        chains = sorted(chains, key=sorter)
        worker(chains[1:])
        # chains = sorted(chains)
        # with Pool(processes=2) as pool:
        #     pool.map(worker, chains, chunksize=len(chains)//5)
        # for ch in tqdm(chains, desc="Chains"):
        #     Similarity[ch] = {}
        #     for root, hotels, ___ in list(os.walk(os.path.join("features2", ch))):
        #         hotels = sorted(hotels)
        #         hotel_self_similarity = None
        #         hotel_vector = torch.zeros((2048)).to(device)
        #         for h in hotels:
        #             rooms = imgs_for_hotel(os.path.join(root, h))
        #             pairs = get_pairs(rooms)
        #             for r in rooms.keys():
        #                 if len(rooms[r]) == 3:
        #                     hotel_vector += whole_image_avg(torch.load(rooms[r][2]))
        #             if not pairs:
        #                 continue
        #             p_c = cos_sim(pairs, rooms)
        #             # print("Chain: {}; ID: {}; AVG Sim: {}".format(ch, h, hotel_self_similarity))
        #             hotel_self_similarity = sum(p_c.values()) / len(p_c.values())
        #             # hist = calculate_hist(p_c)
        #             Similarity[ch][h] = (hotel_self_similarity,hotel_vector / len(rooms.keys()))
        #         for k in Similarity.keys():
        #             if Similarity.values():
        #                 calulate_hist_per_room(Similarity[k], k)
        #                 calculate_hist_same_room(Similarity[k], k)
        #         Similarity = {}
                    

def main():
    comp()

    # for k in tqdm(data.keys(), desc="Calculating Histograms"):
    #     if data.values():
    #         # calulate_hist_per_room(data[k], k)
    #         calculate_hist_same_room(data[k], k)

# %%
if __name__ == "__main__":
    main()

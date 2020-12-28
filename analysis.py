import multiprocessing
import os
import pickle
from glob import glob
import torch
import itertools
import random
from tqdm import tqdm
import plotly.figure_factory as ff
from multiprocessing import Pool


class Analysis:
    restrictions = [1,74,76,65,41]
    hotels = {}

    self_sim = []
    diff_sim = []

    used_chains = []

    def __init__(self, path):
        chains = glob(path + "/*/")
        for chain in chains:
            hotels = sorted(glob(chain + "/*/"), key=self.sorter)
            self.hotels[chain.split("/")[-2]] = hotels
        self.device = torch.device('cuda:3')
            
    def compute_avg_diff(self):
        if os.path.exists("vectors/same.pckl"):
            with open("vectors/same.pckl", "rb") as p1:
                self.self_sim = pickle.load(p1)
        else:
            self.same_hotel_similarity()

            with open("vectors/same.pckl", "wb") as p1:
                pickle.dump(self.self_sim, p1)

            self.diff_hotel_similarity()
            self.hists()

            with open("vectors/diff.pckl", "wb") as p2:
                pickle.dump(self.diff_sim, p2)


    def same_hotel_similarity(self):
        for chain in tqdm(self.hotels, desc="Same Hotel Similarity"):

            for hotel in self.hotels[chain]:
                hotels_vectors = glob(hotel + "**/fts.pt", recursive=True)

                pairs = itertools.combinations(hotels_vectors, 2)
                self.pair_cos_distance(pairs, 0)
            
    def diff_hotel_similarity(self):
        n = 5
        for chain in tqdm(self.hotels, desc="Different Hotel Similarity"):
            vectors = glob("features/" + chain + "/**/fts.pt", recursive=True)

            other_vectors = []
            p = multiprocessing.Pool()

            other_vectors = p.map(self.diff_worker, chain, len(self.hotels) // 5)

            p.close()
            p.join()
            self.used_chains = []
            
            pairs = itertools.product(other_vectors[0], vectors)
            self.pair_cos_distance(pairs, 1)

    def diff_worker(self, chain):
        for i, hotel2 in enumerate(self.hotels):
            if hotel2 == chain or hotel2 in self.used_chains:
                continue
            
            other_vectors = glob("features/" + hotel2 + "/**/fts.pt", recursive=True) 
            self.used_chains.append(hotel2)
            return other_vectors

    def get_random_chain(self, current_key):
        for ch in self.hotels:
            if ch != current_key and ch not in self.used_chains:
                self.used_chains.append(ch)
                return ch


    def pair_cos_distance(self, pairs, which):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for pair in pairs:
            v1 = torch.load(pair[0]).to(self.device); v2 = torch.load(pair[1]).to(self.device)
            if which == 0:
                self.self_sim.append(float(cos(v1,v2)))
            elif which == 1:
                self.diff_sim.append(float(cos(v1,v2)))
            else:
                print("Invalid tpye for cos dis")

    def hists(self):
        # n = 500
        # d1 = random.choices(self.self_sim, k=n)
        # d2 = random.choices(self.diff_sim, k=n)
        labels = ['Same Hotel', 'Diff Hotel']
        fig = ff.create_distplot([self.self_sim, self.diff_sim], group_labels=labels, show_hist=False)
        fig.layout.update({'title': 'Similarities Between Same and Different Hotels'})
        fig.write_image("combined_hist2.png")


        fig = ff.create_distplot([self.self_sim], group_labels=[labels[0]], show_hist=False)
        fig.layout.update({'title': 'Similarities Between Same Hotels'})
        fig.write_image("same_hist2.png")

        fig = ff.create_distplot([self.diff_sim], group_labels=[labels[1]], show_hist=False)
        fig.layout.update({'title': 'Similarities Between Different Hotels'})
        fig.write_image("diff_hist2.png")

    def sorter(self, x):
        return int(x.split("/")[-2])




if __name__ == "__main__":
    a = Analysis("features")
    a.compute_avg_diff()
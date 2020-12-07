import os
import pickle
from glob import glob
import torch
import itertools
import random
from tqdm import tqdm
import plotly.figure_factory as ff

class Analysis:
    restrictions = [1,74,76,65,41]
    hotels = {}

    self_sim = []
    diff_sim = []

    def __init__(self):
        chains = glob("features/*/")
        for chain in chains:
            hotels = sorted(glob(chain + "/*/"), key=self.sorter)
            self.hotels[chain.split("/")[-2]] = hotels
        self.device = torch.device('cuda:3')
            
    def compute(self):
        self.same_hotel_similarity()
        self.diff_hotel_similarity()
        self.hists()

    def same_hotel_similarity(self):
        hotels_vectors = []

        for chain in tqdm(self.hotels, desc="Same Hotel Similarity"):
            for hotel in self.hotels[self.get_random_chain(chain)]:
                hotels_vectors = glob(hotel + "**/fts.pt", recursive=True)

                pairs = itertools.combinations(hotels_vectors, 2)
                self.pair_cos_distance(pairs, 0)
                
    def diff_hotel_similarity(self):
        n = 10
        for chain in tqdm(self.hotels, desc="Different Hotel Similarity"):
            for hotel in self.hotels[chain]:
                vectors = glob(hotel + "**/fts.pt", recursive=True)

                other_vectors = []
                for hotel in self.hotels[self.get_random_chain(chain)]:
                    other_vectors = glob(hotel + "**/fts.pt", recursive=True) 
            
                pairs = itertools.product(other_vectors, vectors)
                self.pair_cos_distance(pairs, 1)

               
    def get_hotels(self, chain):
        for hotel in self.hotels[chain]:
            vectors = glob(hotel + "**/fts.pt", recursive=True)
            return vectors

    def get_random_chain(self, current_key):
        other_hotel = None
        while other_hotel is None:
            rand = random.choice(list(self.hotels))
            if rand != current_key:
                return rand


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
        labels = ['Same Hotel', 'Diff Hotel']
        fig = ff.create_distplot([self.self_sim, self.diff_sim], group_labels=labels, show_hist=False)
        fig.layout.update({'title': 'Similarities Between Same and Different Hotels'})
        fig.write_image("combined_hist.png")

    def sorter(self, x):
        return int(x.split("/")[-2])




if __name__ == "__main__":
    anal = Analysis()
    anal.compute()
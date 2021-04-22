import pandas as pd
from scipy.spatial import distance
import torch
from torch import nn
import features
import pickle
class Index:
    def __init__(self) -> None:
        self.df = pd.read_csv("data/train/features.csv")
        # df = df.loc[df[1] == feature]
class kNN:
    def __init__(self, k) -> None:
        # self.comb_index = Index()
        self.comb_index = pd.read_csv("data/train/features.csv", names=["Idx", "Class", "Path"]).unique()
        self.k = k
    

    def search(self, query_img, acc_classes):
        closest = []
        
        for idx, acc in enumerate(acc_classes):
            index = self.comb_index.loc[self.comb_index["Class"] == acc]
            
            closest.append(self.knn(index, query_img))
        return closest[:k]
    
    def knn(self, index, query_img):
        closest_img = None
        closest_distance = 0

        for idx, row in index.iterrows():
            path = row["Path"]+"/indiv_features1.pt"
            if path.split("/")[-1] == features.path.split("/")[-1]: continue

            # other = torch.unsqueeze(torch.load(path)[row["Idx"]], 0).to('cuda:{}'.format(features.gpu))
            other = self.__load_pckl(path)[row["Idx"]]
            
            # dist = float(cos(other, query_img))
            dist = distance.cosine(query_img, other)
            if dist > closest_distance:
                closest_distance = dist
                closest_img = row["Path"]
        return closest_img, closest_distance

    def evaluate_accuracy(self, query_index):
        for idx, row in query_index.iterrows():
            query_image = torch.load(row["Path"]+"/indiv_features1.pt")[row["Idx"]]

            search(query_image,)
            
    def __load_pckl(self, path):
        with open(path, "rb") as f:
            ret = pickle.load(f)
        return ret
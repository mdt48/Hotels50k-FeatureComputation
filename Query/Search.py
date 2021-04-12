import pandas as pd
import torch
import features

class Index:
    def __init__(self) -> None:
        self.df = pd.read_csv("data/train/features.csv")
        # df = df.loc[df[1] == feature]
class kNN:
    def __init__(self) -> None:
        # self.comb_index = Index()
        self.comb_index = pd.read_csv("data/train/features.csv", names=["Idx", "Class", "Path"])
        
    
    def search(self, query_img, acc_classes):
        closest = []
        query_img.to('cuda:{}'.format(features.gpu))
        for idx, acc in enumerate(acc_classes):
            index = self.comb_index.loc[self.comb_index["Class"] == acc]
            closest.append(self.knn(index, torch.unsqueeze(query_img[idx], 0)))
        return closest

    def knn(self, index, query_img):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        closest_img = None
        closest_distance = 0

        for idx, row in index.iterrows():
            path = row["Path"]+"/indiv_features1.pt"
            if path.split("/")[-1] == features.path.split("/")[-1]: continue

            other = torch.unsqueeze(torch.load(path)[row["Idx"]], 0).to('cuda:{}'.format(features.gpu))
            
            dist = float(cos(other, query_img))

            if dist > closest_distance:
                closest_distance = dist
                closest_img = row["Path"]
        return closest_img
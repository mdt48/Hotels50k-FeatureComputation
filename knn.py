import json
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from tqdm import tqdm, trange

from analysis import load_pckl

random.seed(100)



def build_index(x):
    """
    Build an np index at run time given the dataframe x.
    """
    res = np.zeros( (len(x), 2048), dtype="float16" )
    c = 0
    for idx, row in x.iterrows():
        
        res[c] = torch.load(row["Path"]+"/indiv_features.pt")[int(row["Idx"])].numpy()
            
        c+= 1
    return res 


def avg(l):
    """Helper function to return the average of a list of numbers"""
    return sum(l)/len(l)


def my_distance(x, y):
    """Custom distance function for testing"""
    return distance.cosine(x, y)


def knn(n, paths, train_idx, X_test, X_test_csv, mode):
    """ 
    Fit and evaluate a kNN model with n=n for all images in path.
    train_idx: the root path of the train dataset
    X_test: dataframe containing test values
    paths: all the images to test
    """
    results = []

    t = trange(len(paths), desc="K={}: 0.0".format(n), leave=True)

    # load previous results from file if continuing experiment
    try:
        with open("results/knn/FULL-{}.json".format(mode+ "-"+ str(n)), "r") as f:
            results = list(json.loads(f.read())[0])
    except FileNotFoundError:
        pass
    
    # For each image...
    for i in t:
        path = paths[i]

        # get the image, and the corresponding values of chain/instance for evaluation
        row = X_test_csv.loc[X_test_csv["Path"] == path]
        classes = row["Class"].tolist()
        Y_test = row[mode].tolist()

        # for every class(instance of path)
        for idx, cl in enumerate(classes):
            out = []

            # load the index(precomputed) for each class from the train set
            X = load_pckl("data/{}/indices/{}.pckl".format(train_idx, cl))
            X_csv = pd.read_csv("data/train-final/indices/{}.csv".format(cl), names=["Idx", "Class", "Path"])
            X_csv[mode] = [int(row.split("/")[2 if mode.lower() == "chain" else 3]) for row in X_csv["Path"]]
            Y = X_csv[mode].tolist()
            
            # can do a knn of size n if there are not n samples
            if len(X) < n:
                n = len(X)

            # fit knn model, brute force
            KNN = KNeighborsClassifier(n_neighbors=n)
            KNN.fit(X, Y)

            # get the nearest neighbors
            res = KNN.kneighbors( X_test[int(row.index[idx])].reshape(1, -1) )

            # evaluate results
            for i in res[1]:
                for j in i:
                    out.append(X_csv.iloc[j][mode])

            results.append( round( out.count(Y_test[0])/(n), 5)) 

        t.set_description( "K={}: {}".format(n, round(avg(results),5) ) )

        # checkpoint results
        with open("results/knn/FULL-{}.json".format(mode+ "-"+ str(n)), "w") as f:
            json.dump([results, round(avg(results), 3) ], f)
    return avg(results)

def main(train_index, test_index, k_values, mode):
    X_test_csv = pd.read_csv("data/{}/features.csv".format(test_index), names=["Idx", "Class", "Path"], low_memory=False)
    X_test_csv[mode] = [int(row.split("/")[2 if mode.lower() == "chain" else 3]) for row in X_test_csv["Path"]]
    X_test = build_index(X_test_csv)

    paths = X_test_csv["Path"].unique().tolist()

    for k in k_values:
        knn(k, paths, train_index, X_test, X_test_csv, mode)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("INVALID NUMBER OF ARGUMENTS\nPLEASE INPUT IN FORM 'python knn.py [train index] [test index] [k values-1,2,...] [chain or instance]")
    tr_idx = sys.argv[1]
    te_idx = sys.argv[2]
    k = [int(x) for x in sys.argv[3].split(",")]
    mode = sys.argv[4]
    
    print("Beginning kNN:\n\tTraining set: {}\n\tTesting set: {}\n\tChain/Instance: {}\n\tK values: {}".format(tr_idx, te_idx, mode, k))
    main(tr_idx, te_idx, k, mode)

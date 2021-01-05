import argparse
import faiss
from faiss.swigfaiss_avx2 import Index
import torch
from glob import glob
import numpy as np
from tqdm import tqdm
import sys, pickle, os
from query import index as I

def reset_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def create_index():
    print("\n\nLoading vectors 🔨")
    # hotels = sorted(glob("features" + "/**/fts.pt", recursive=True), key=lambda x: x.split("/")[1])
    hotels = sorted([f for f in glob("/pless_nfs/home/mdt_/Hotels50-FeatureComputation/data/imgs/test_indexing" + "/**/fts.pt", recursive=True)  if "-1" not in f], key=lambda x: x.split("/")[1])

    print("Loading vectors ✅\n\n")

    nb = len(hotels)
    d = 2048
    vectors = np.ones((nb, d),dtype="float32")

    print("Compiling vectors 🔨")
    
    if os.path.exists("data/globbed_index.npy"):
        with open("data/globbed_index.npy", "rb") as tcam:
            vectors = np.load(tcam)
    else:
        for idx, hotel in tqdm(enumerate(hotels)):
            vector = torch.load(hotel, map_location='cuda:0').cpu().numpy().astype('float32')
            vectors[idx] = vector[0]
            
        with open("data/globbed_index.npy", "wb") as tcam:
            np.save(tcam, vectors)

    # with tqdm(total=nb) as pbar:

    
            # pbar.update(1)
    faiss.normalize_L2(vectors)

    print("Compiling vectors ✅\n\n")
    # quantizer = faiss.IndexFlatIP(d)
    # index = faiss.IndexIVFFlat(quantizer, d, 100)
    
    # index = faiss.IndexLSH(d, 2*d)
    # index.train(vectors)
    # index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
    # index = faiss.IndexFlatIP(d)
    index = faiss.IndexHNSWFlat(d, 32)
    
    # index = faiss.GpuIndexFlat(d)
    # index = faiss.IndexHNSWFlat(d, 32)
    # index = faiss.index_factory(d, "IVF400_HNSW5,PQ64")
    # index.train(vectors)
    index.train(vectors)
    index.add(vectors)

    return hotels, index

def get_query(path):
    # return sorted(glob(path + "/**/fts.pt", recursive=True), key=lambda x: x.split("/")[1])
    return sorted([f for f in glob("features" + "/**/fts.pt", recursive=True)  if "-1" not in f], key=lambda x: x.split("/")[1])


def k_nearest(hotel, hotels, index):
    k_chain = 5
    k_instance = 100 
    query = np.ones((1, 2048),dtype="float32")
    v = torch.load(hotel, map_location='cuda:0').cpu().numpy().astype('float32')
    faiss.normalize_L2(v)
    query[0] = v

    D_ch, I_ch = index.search(query, k_chain)
    D_in, I_in = index.search(query, k_instance)


    t = hotel.split("/")
    # print("🔝 k={} closest to {}_{}_{}".format(k,t[1], t[2], t[4]))
    ch_c = 0
    in_c = 0

    for i in I_ch[0]:
        ch = hotels[int(i)].split("/")[8]
        if t[1] == ch:
            ch_c += 1
    
    for i in I_in[0]:
        

        if t[2] == hotels[int(i)].split("/")[9]:
            in_c += 1
        # print("{} {}: {}_{}_{}".format(status, idx, ids[1], ids[2], ids[3]))
    # print("\n\n")
    return ch_c / k_chain, in_c / k_instance
    print("\n\nTop k={} accuracy={}".format(k, counter / k))

def main(path):  
    hotels, index = create_index()
    query_hotels = get_query(args.path)
    c_total = 0; in_total = 0
    total = len(query_hotels)
    for hotel in tqdm(query_hotels):
        result = k_nearest(hotel, hotels, index)
        c_total += result[0]; in_total += result[1]
    print("\n\nTop k={} chain accuracy={}".format(5, c_total / total))
    print("\n\nTop k={} instance accuracy={}".format(100, in_total / total))


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='End to End Query')
    parser.add_argument('--Q', type=str,
                    help='Path to Image you would like to query', default="/pless_nfs/home/mdt_/Hotels50-FeatureComputation/features")
    parser.add_argument('--I', type=str,
                    help='Path to folder from which to build Index', default="data/imgs/test_indexing")
    parser.add_argument('--M', type=str,
                    help='Indexing methods: 0 for flat L2, 1 for flat IP, 2 HSNW Flat', default="0")
    
    args = parser.parse_args()

    I.Index(args)
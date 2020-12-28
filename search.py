from typing import Counter
import faiss
from numpy.core.defchararray import index
import torch
from glob import glob
import numpy as np
from tqdm import tqdm
import sys

def reset_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def create_index():
    print("\n\nLoading vectors 🔨")
    hotels = sorted(glob("features" + "/**/fts.pt", recursive=True), key=lambda x: x.split("/")[1])

    print("Loading vectors ✅\n\n")

    nb = len(hotels)
    d = 2048
    vectors = np.ones((nb, d),dtype="float32")

    print("Compiling vectors 🔨")
    # with tqdm(total=nb) as pbar:
    for idx, hotel in tqdm(enumerate(hotels)):
        vector = torch.load(hotel).cpu().numpy().astype('float32')
        vectors[idx] = vector[0]
            # pbar.update(1)
    # faiss.normalize_L2(vectors)

    print("Compiling vectors ✅\n\n")
    # quantizer = faiss.IndexFlatIP(d)
    # index = faiss.IndexIVFFlat(quantizer, d, 508, faiss.METRIC_INNER_PRODUCT)
    # index = faiss.IndexLSH(d, 2*d)
    # index.train(vectors)
    # index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
    # index = faiss.IndexFlatIP(d)
    index = faiss.IndexHNSWFlat(d, 32)
    # index = faiss.index_factory(d, "IVF400_HNSW5,PQ64")
    # index.train(vectors)
    index.add(vectors)

    return hotels, index

def get_query():
    return sorted(glob("test_data" + "/**/fts.pt", recursive=True), key=lambda x: x.split("/")[1])

def k_nearest(hotel, hotels, index):
    k = 1
    query = np.ones((1, 2048),dtype="float32")
    query[0] = torch.load(hotel).cpu().numpy().astype('float32')
    D, I = index.search(query, k)

    t = hotel.split("/")
    print("🔝 k={} closest to {}_{}_{}".format(k,t[1], t[2], t[4]))
    ch_c = 0
    in_c = 0
    for idx, i in enumerate(I[0]):
        ids = hotels[i].split("/")

        status = None
        if ids[1] == hotel.split("/")[1]:
            ch_c += 1
            status = "💲"
        else:
            status = "💢"


        if ids[-2] == hotel.split("/")[-2]:
            in_c += 1
        print("{} {}: {}_{}_{}".format(status, idx, ids[1], ids[2], ids[3]))
    # print("\n\n")
    return ch_c / k, in_c / k
    print("\n\nTop k={} accuracy={}".format(k, counter / k))

def main():
    hotels, index = create_index()
    query_hotels = get_query()
    # query_hotels = hotels.copy()
    c_total = 0; in_total = 0
    for hotel in tqdm(query_hotels):
        result = k_nearest(hotel, hotels, index)
        c_total += result[0]; in_total += result[1]
    print("\n\nTop k={} chain accuracy={}".format(5, c_total / len(query_hotels)))
    print("\n\nTop k={} instance accuracy={}".format(5, in_total / len(hotels)))


if __name__ == "__main__":
    main()
import pickle
import faiss
import torch
import os
import csv
from tqdm import tqdm
import numpy as np
from glob import glob
import features

class QueryBuilder():
    def __init__(self, args,new_img=0) -> None:
        super().__init__()
        self.args = args
        self.new = new_img
        self.mode = args.SorM.lower()

    def build(self):
        if self.mode == "s":
            return SingleIndexQuery(self.args, self.new)
        elif self.mode == "m":
            return MultiIndexQuery(self.args, self.new)
        else:
            raise ValueError("Expected 'S' or 'M' for building Index")

class SingleIndexQuery:
    def __init__(self, args, new_img=0):
        self.query_path = args.imgs
        self.index_path = args.index
        self.new = new_img
        self.index_method = args.method
        self.mode = args.SorM.lower()
    
    def build_and_search(self):
        self.__build()
        self.__search()

    def __build(self):
        self.query_imgs = self.__get_files(self.query_path)

        # See if the index has already been created!
        self.__check_index_exists()
        print(len(self.index_imgs))
        # self.index_imgs = self.__get_files(self.index_path)
        # self.index = self.build_index(self.index_method)
        self.query = self.__build_query()

    def __get_files(self, path):
        if os.path.isfile(path):
            return [path]
        return sorted([f for f in glob(path + "/**/fts.pt", recursive=True)  if "-1" not in f], key=lambda x: x.split("/")[1])

    def build_index(self, type, use_index=None):
        type = int(type)

        nb = len(self.index_imgs)
        d = 2048

        index_vectors = np.ones((nb, d),dtype="float32")
        
        self.__add_vectors(self.index_imgs, index_vectors, 0, use_index)

        index = None
        if type == 0:
            index = faiss.IndexFlatL2(d)
            
        elif type == 1:
            faiss.normalize_L2(index_vectors)

            index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
            # index = faiss.IndexFlatIP(d)
            

        elif type == 2:
            index = faiss.IndexHNSWFlat(d, 32)
        elif type == 3:
            index = faiss.IndexPQ(d)

        
        # index.train(index_vectors)

        index.add(index_vectors)
        
        return index

    def __build_query(self):
        nb = len(self.query_imgs)
        q_vectors = self.query_imgs

        if self.new == 1:
            nb = 1
            q_vectors = features.feat_2048

        query = np.ones((nb, 2048),dtype="float32")
        self.__add_vectors(q_vectors, query, 0)
        if self.index_method == 1:
            faiss.normalize_L2(query)

        # query[0] = features.feat_2048.cpu().numpy().astype('float32')
        return query

    def __add_vectors(self, src, dst, norm, use_index=None):
        for idx, v in enumerate(src):
            if use_index:
                vec = torch.load(v[0], map_location='cuda:0').cpu().numpy().astype('float32')
                vec = vec[v[1]]
                dst[idx] = vec
            else:
                vec = torch.load(v, map_location='cuda:0').cpu().numpy().astype('float32')
                dst[idx] = vec[0]
    
    def __search(self):
        print("Searching")
        k_chain = 5
        k_instance = 100 

        D_ch, I_ch = self.index.search(self.query, k_chain)
        if self.query_path == "/pless_nfs/home/mdt_/Hotels50-FeatureComputation/features" or self.query_path == "data/imgs/test_indexing/":
            D_in, I_in = self.index.search(self.query, k_instance)

            ch_total = 0
            for idx, res in enumerate(I_ch):
                ch_c = 0
                chain = self.query_imgs[idx].split("/")[-4]
                for i in res:
                    match = self.index_imgs[i].split("/")[-4]

                    if chain == match:
                        ch_c += 1
                ch_total += ch_c / k_chain
            
            in_total = 0
            for idx, res in enumerate(I_in):
                ch_i = 0
                chain = self.query_imgs[idx].split("/")[-4]
                for i in res:
                    match = self.index_imgs[i].split("/")[-4]

                    if chain == match:
                        ch_i += 1
                in_total += ch_i / k_instance
            
            print("Chain top 5: {:.2f}%".format((ch_total / len(self.query_imgs))*100))
            print("Instance top 100: {:.2f}%".format((in_total / len(self.query_imgs)*100)))
        else:
            for i in I_ch[0]:
                print(self.index_imgs[i])
    
    def __check_index_exists(self):
        self.__img_list_serialization()

        print("Getting Index")
        if self.mode == "s":
            if os.path.exists("data/indexes/SingleIndex.fss"):
                self.__index_deserialization()
                return             
        elif self.mode == "m":
            if os.path.exists("data/indexes/MultiIndex.fss"):
                self.__index_deserialization()
                return 

        self.index = self.build_index(self.index_method)
            
        
        self.__index_serialization()

    def __index_serialization(self):
        if self.mode == "s":
            faiss.write_index(self.index, "data/indexes/SingleIndex.fss")
        else:
            faiss.write_index(self.index, "data/indexes/MultiIndex.fss")
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def __index_deserialization(self):
        if self.mode == "s":
            # with open("data/indexes/SingleIndex.fss", "rb") as index_file:
                # faiss.write_index(self.index, index_file)
            self.index = faiss.read_index("data/indexes/SingleIndex.fss")
        else:
            self.index = faiss.write_index(self.index, "data/indexes/MultiIndex.fss")

    def __img_list_serialization(self):
        if os.path.exists("data/indexes/img_list.pckl"):
            with open("data/indexes/img_list.pckl", "rb") as imgs:
                self.index_imgs = pickle.load(imgs)
        else:
            self.index_imgs = self.__get_files(self.index_path)
            print("Got File Names: {}".format(len(self.index_imgs)))
            with open("data/indexes/img_list.pckl", "wb") as imgs:
                pickle.dump(self.index_imgs, imgs)

        

class MultiIndexQuery(SingleIndexQuery):
    def __init__(self, args, new_img=0):
        super().__init__(args, new_img=new_img)
        self.index_imgs = []
    
    def build_and_search(self):
        self.__build_class_dict()
        self.__build_multi_index()
        self.__search()

    def __build_class_dict(self):
        self.class_dict = {i:[] for i in range(1, 151)}

        chains = self.__get_subdirs(self.index_path)

        for chain in tqdm(chains[:20], desc="Collecting Vectors"):
            instance = self.__get_subdirs(chain)

            for inst in instance:
                rooms = self.__get_subdirs(inst)

                for room in rooms:
                    files = self.__get_files(room)

                    with open(files[0], 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        next(reader)

                        for i, row in enumerate(reader):
                            self.class_dict[int(row[0])].append( (files[1], i) )

    def __build_multi_index(self):
        self.indicies = {i for i in range(1, 151)}
        for cl in tqdm(self.class_dict.keys(), desc="Indexing"):
            if not self.class_dict[cl]:
                self.class_dict[cl] = None
                continue

            
            self.index_imgs = self.class_dict[cl]

            self.index = self.build_index(self.index_method, use_index=1)
        

    def __get_subdirs(self, p):
        return glob(p + "/*/")

    def __get_files(self, p):
        return glob(p + "/*.*")
    
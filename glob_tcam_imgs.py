import pickle
import os
import glob
files = []
for filename in glob.iglob('/pless_nfs/home/datasets/Hotels-50K/images/test/**/*.jpg', recursive=True):
    files.append(os.path.abspath(filename))
print(len(files))
with open("/pless_nfs/home/mdt_/Hotels50-FeatureComputation/test_images.pckl", "wb") as f:
    pickle.dump(files, f)
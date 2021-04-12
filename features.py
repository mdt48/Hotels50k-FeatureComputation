import torch
import torch.nn as nn
import os
import numpy as np
from scipy.interpolate import interp1d
from glob import glob
from lib.utils import as_numpy
import csv

gpu = 0
feat_2048 = torch.zeros(size=(1, 2048), device=torch.device('cuda:{}'.format(gpu)))
feat_2048_whole = torch.zeros(size=(1, 2048, 38, 50), device=torch.device('cuda:{}'.format(gpu)))
feat_150 = torch.Tensor



def compute_image_representation(feat_150, feat_2048_whole, path, save=True):
    cuda0 = torch.device('cuda:{}'.format(gpu))
    
    num_classes = 150
    # size of each feat
    size_2048 = feat_2048_whole.size()
    size_150 = feat_150.size()

    # features = torch.zeros((size_162[0], size_162[1]))

    # most probable class at each pixel
    _, feat_150_pred = torch.max(feat_150, dim=1)
    feat_150_pred = as_numpy(feat_150_pred.squeeze(0))

    accepted_classes = [0 for i in range(num_classes)]

    # find corresponding coords
    x = nn.functional.interpolate(torch.from_numpy(feat_150_pred).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor), size=(size_2048[2], size_2048[3]), mode='bilinear', align_corners=False)
    x = x.squeeze().squeeze().type(torch.int64)

    #get features with more than 10% of pixels
    features_coords = torch.zeros((size_150[2], size_150[3], 1))

    threshold = 0.1; pix = x.size()[0] * x.size()[1]
    accepted_classes = [i+1 for i in range(len(accepted_classes)) if (len(np.argwhere(feat_150_pred == i+1)) / pix) > threshold]

    # # calculate average for each class in 2048d
    feat_2048 = feat_2048_whole.permute((0,2,3,1)).type(torch.cuda.HalfTensor)
    img_rep = torch.zeros((len(accepted_classes), 2048),device=cuda0).type(torch.cuda.HalfTensor)
    for idx, cl in enumerate(accepted_classes):
        coords = np.argwhere(x.numpy() == accepted_classes[accepted_classes.index(cl)])
        for coord in coords:
            img_rep[idx] = img_rep[idx].add(feat_2048[0][coord[0]][coord[1]])
        img_rep[idx] = torch.div(img_rep[idx], len(coords))
    if save:
        torch.save(img_rep, os.path.join(path, "indiv_features1.pt"))
        to_csv(accepted_classes, path)
    else:
        return img_rep, accepted_classes

def to_csv(accepted_classes, img_path):
    """
    Write new row in form of index, class, path
    """
    with open('data/train/features.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(accepted_classes)):
            writer.writerow([i, accepted_classes[i], img_path])
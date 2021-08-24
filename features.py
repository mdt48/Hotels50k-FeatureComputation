import torch
import torch.nn as nn
import os
import numpy as np
from scipy.interpolate import interp1d
from glob import glob
from lib.utils import as_numpy
import csv
import pickle

# temp til better solution
fc_feature = None
pred_feature = None


class Feature:
    def __init__(self, fc_feature=fc_feature, pred_feature=pred_feature, device=0, num_classes=150, save_path=None, threshold=0.1):
        self.device = torch.device('cuda:{}'.format(device))
        self.num_classes = num_classes
        self.save_path = save_path
        self.threshhold = threshold

        self.fc_feat = fc_feature.to(self.device).type(
                torch.cuda.HalfTensor)
        self.pred_feat = pred_feature.to(self.device).type(
                torch.cuda.HalfTensor)

        self.fc_feat_size = fc_feature.shape
        self.fc_dim = fc_feature.shape[-1]

        self.pred_feat_size = pred_feature.shape

    def compute_image_rep(self):
        interp_feat = self.interpolate_feature(
            self.pred_feat, (self.fc_feat_size[1], self.fc_feat_size[2]))
        accepted_classes = self.get_accepted_classes(interp_feat)
        computed_features = self.pool_feature(accepted_classes, interp_feat)

    def pool_feature(self, accepted_classes, interp_feat):
        for idx, acc in enumerate(accepted_classes):
            locations = np.argwhere(
                interp_feat.numpy() == accepted_classes[accepted_classes.index(acc)])

            img_rep = torch.zeros((len(accepted_classes), self.fc_dim), device=self.device).type(
                torch.cuda.HalfTensor)
            for loc in locations:
                img_rep = img_rep[idx].add(self.fc_feat[0][loc[0]][loc[1]])
            img_rep[idx] = torch.div(img_rep[idx], len(locations))
        return img_rep

    def get_accepted_classes(self, feature):
        total_pixels = self.fc_feat_size[1] * self.fc_feat_size[2]

        accepted_classes = []
        for i in range(self.num_classes):
            if (len(np.argwhere(feature.numpy() == i+1)) / total_pixels) > self.threshhold:
                accepted_classes.append(i+1)
        return accepted_classes

    def interpolate_feature(self, feature, size):
        _, feat_150_pred = torch.max(feature, dim=1)
        feat_150_pred = as_numpy(feat_150_pred.squeeze(0))

        interp_feature = nn.functional.interpolate(torch.from_numpy(feat_150_pred).unsqueeze(
            0).unsqueeze(0).type(torch.FloatTensor), size=size, mode='bilinear', align_corners=False)
        interp_feature = interp_feature.squeeze().squeeze().type(torch.int64)

        return interp_feature

    # TODO: SAVE THE FEATURES
    def save_pooled_features(self, image_rep):
        pass
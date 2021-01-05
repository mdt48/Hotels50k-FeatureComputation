# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our data
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode, find_recursive, setup_logger, compute
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from config import cfg
import features, random
from Query import SingleIndexQuery, MultiIndexQuery, QueryBuilder



# colors = loadmat('data/color150.mat')['colors']
# names = {}
# with open('data/object150_info.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         names[int(row[0])] = row[5].split(";")[0]
# colors = []
# names = {}
# from extras import colors
# colors_names = colors.getColors()
# for idx, pair in enumerate(colors_names):
#     colors.append(pair[0])
#     names[idx+2] = pair[1]

import pickle
# col = getColors()
colors = []
names = {}
with open("Colors/colors_Hotels_and_ADE_162.pckl", "rb") as p:
    data = pickle.load(p)
    for idx, d in enumerate(data):
        # colors.append(np.array(d[0], np.uint8))
        names[idx+1] = d
    colors = np.array(data, np.uint8)

def sample(file_list, N):
    sample = []

    for i, line in enumerate(file_list):
        if i < N:
            sample.append(line)
        elif i >= N and random.random() < N / float(i+1):
            replace = random.randint(0, len(sample)-1)
            sample[replace] = line

    return sample

def visualize_result(data, pred, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(features.path, img_name.replace('.jpg', '.png')))


def test(segmentation_module, loader, gpu):
    segmentation_module.eval()

    # pbar = tqdm(total=len(loader))
    for batch_data in tqdm(loader):
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']


        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
        # # torch.save(features.feat_2048, os.path.join(features.path, "fts.pt"))
        # a = compute(features.feat_150, features.feat_2048_whole, features.path)
        # # torch.save(a, features.path + "/fts.pt")
        # torch.save(features.feat_2048, features.path + "/fts.pt")

        # visualization
 
        # visualize_result(
        #     (batch_data['img_ori'], batch_data['info']),
        #     pred,
        #     cfg
        # )

        # pbar.update(1)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)
    segmentation_module.cuda()

    # Main loop
    test(segmentation_module, loader_test, gpu)
    
    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="an image paths, or a directory name for images to be queried"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )

    parser.add_argument('--index', type=str,
                    help='Path to folder from which to build Index', default="/pless_nfs/home/mdt_/Hotels50-FeatureComputation/data/index")

    parser.add_argument('--method', type=str,
                    help='Path to folder from which to build Index', default="1")
    parser.add_argument('--testing', type=str,
                    help='Run Analysis?', default=None)    
    parser.add_argument('--SorM', type=str,
                    help='Run Analysis?', default=None)                                  
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    if args.testing:
        args.imgs = "data/imgs/test_indexing/"
        args.index = "/pless_nfs/home/mdt_/Hotels50-FeatureComputation/data/index"

    # new image that must be segmented
    new_img = 0
    if os.path.isfile(args.imgs):
        new_img = 1
        cfg.merge_from_file(args.cfg)
        cfg.merge_from_list(args.opts)
        # cfg.freeze()

        logger = setup_logger(distributed_rank=0)   # TODO
        logger.info("Loaded configuration file {}".format(args.cfg))
        logger.info("Running with config:\n{}".format(cfg))

        cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
        cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

        # absolute paths of model weights
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

        # generate testing image list
        # print("IMAGES:" + args.imgs)
        # if os.path.isdir(args.imgs[0]):
        #     print("images[0]" + args.imgs)
        #     imgs = find_recursive(args.imgs)
        # else:
        #     imgs = [args.imgs]

        if os.path.isfile(args.imgs):
            if args.imgs.endswith(".pckl"):
                with open(args.imgs, "rb") as pckl:
                    imgs = pickle.load(pckl)
            else:
                imgs = [args.imgs]
        else:
            imgs = find_recursive(args.imgs)

        # imgs = sample(imgs, 200000)

        assert len(imgs), "imgs should be a path to image (.jpg) or directory."

        cfg.list_test = [{'fpath_img': x} for x in imgs]

        if not os.path.isdir(cfg.TEST.result):
            os.makedirs(cfg.TEST.result)

        main(cfg, args.gpu)
        # Search(args, new_img=1)
    
    Query = QueryBuilder(args, new_img).build()
    Query.build_and_search()
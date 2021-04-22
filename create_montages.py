from imutils import build_montages
from imutils import paths
import argparse
import random
import cv2
import glob2

imagePaths = list(glob2.glob("data/figures/test_non_torch_take_2/**/*.png"))
random.shuffle(imagePaths)


# initialize the list of images
images = []
# loop over the list of image paths
for imagePath in imagePaths:
	# load the image and update the list of images
	image = cv2.imread(imagePath)
	images.append(image)
# construct the montages for the images
montages = build_montages(images, (300, 400), (9, 8))

cv2.imwrite("montage.png", montages[0])
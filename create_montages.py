from imutils import build_montages
from imutils import paths
import argparse
import random
import cv2
import glob2

imagePaths = list(glob2.glob("data/figures/test_non_torch-take-5/**/*.png"))
print(len(imagePaths))
random.shuffle(imagePaths)


# initialize the list of images
images = []
# loop over the list of image paths
for imagePath in imagePaths:
	# load the image and update the list of images
	image = cv2.imread(imagePath)
	images.append(image)
# construct the montages for the images
montages = build_montages(images, (400, 600), (11, 3))


cv2.imwrite("data/figures/test_non_torch-take-5/test_nt-t5.png", montages[0])

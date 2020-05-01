from skimage import io, color
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

class point:
	def __init__(self, pos, orb, knn_list):
		self.pos = pos # position information of key point
		self.orb = orb # orb feature
		self.knn_list = knn_list
		
class homography:
     def __init__(self, H, img_idx):
        self.H = H  # homography matrix
        self.img_idx = img_idx  # match index

def load_data(path):
    files = os.listdir(path)
    l = len(files)
    images = [0] * l
    for file in files:
        idx = int(file[-6:-4]) - 1   # if adobe dataset, remove -1
        images[idx] = color.rgb2grey(io.imread(path + '/' + file))

    return np.array(images)

def get_ORB_feature(images):
	orb = cv2.ORB_create()
	img_pts_dict = {}
	for i in range(len(images)):
		# get orb features of each image
		kp, des = orb.detectAndCompute(images[i], None)

		# draw key points if you want
		# img = cv2.drawKeypoints(images[i], kp, np.array([]))
		# plt.imshow(img)
		# plt.show()

		# store each point's information in one list and map to the image
		pts = []
		for j in range(len(kp)):
			pt = point(kp[j], des[j], [])
			pts.append(pt)
		img_pts_dict[i] = pts
	return img_pts_dict


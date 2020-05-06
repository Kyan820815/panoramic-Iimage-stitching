from preprocesser import preprocesser
from matcher import matcher
from sticher import sticher
from matplotlib import pyplot as plt
import numpy as np
import imutils
import cv2

def main():
	#----------------------------------------------
	#
	# step 1: preprocess data
	#	(1) load data
	#	(2) get feature
	#	(3) find knn for each feature
	#----------------------------------------------
	processer_obj = preprocesser("./data/test5")
	# images = images[:10]
	#----------------------------------------------
	#
	# step 2: find relation between all images
	#	(1) construct candidate map
	#	(2) compute homography matrix
	#----------------------------------------------
	matcher_obj  = matcher(processer_obj.img_pts_dict, 5, 0.7, 4)
	#----------------------------------------------
	#
	# step 3: stich all images and make panorama
	#	(1) select best candidate based on given img
	#	(2) stich each by each
	#	(3) apply ROI on panorama image
	#----------------------------------------------
	stitcher_obj = sticher(matcher_obj)
	pano_img = stitcher_obj.img_pano(processer_obj.images, 0)

	# opencv built-in function for image stitching
	# stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
	# (status, stitched) = stitcher.stitch(images[0:10])

	plt.imshow((pano_img).astype(np.uint8))
	plt.show()

if __name__ == '__main__':
	main()


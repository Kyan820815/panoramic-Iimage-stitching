from preprocesser import preprocesser
from matcher import matcher
from sticher import sticher
from matplotlib import pyplot as plt
import os
import numpy as np
import imutils
import argparse
import cv2

def main(args):
	data_path = data_dir = os.path.dirname(__file__) + "./data/" + args.data
	result_path = data_dir = os.path.dirname(__file__) + "./data/result"
	#----------------------------------------------
	# step 1: preprocess data
	#	(1) load data
	#	(2) get feature
	#	(3) find knn for each feature
	#----------------------------------------------
	processer_obj = preprocesser(data_path)
	#----------------------------------------------
	# step 2: find relation between all images
	#	(1) construct candidate map
	#	(2) compute homography matrix
	#----------------------------------------------
	matcher_obj  = matcher(processer_obj.img_pts_dict, args.candidate, args.lowe_ratio, args.ransac_th)
	#----------------------------------------------
	# step 3: stich all images and make panorama
	#	(1) select best candidate based on given img
	#	(2) stich each by each
	#	(3) apply ROI on panorama image
	#----------------------------------------------
	stitcher_obj = sticher(matcher_obj, args.roi_improve)
	pano_img, roi_pano_img = stitcher_obj.img_pano(processer_obj.images, 0)
	pano_img     = pano_img.astype(np.uint8)
	roi_pano_img = roi_pano_img.astype(np.uint8)
	plt.figure()
	plt.subplot(2, 1, 1)
	plt.imshow(pano_img)
	plt.subplot(2, 1, 2)
	plt.imshow(roi_pano_img)
	plt.show()

	cv2.imwrite(result_path + "/" + args.data + "raw.png", pano_img[...,::-1]) 
	cv2.imwrite(result_path + "/" + args.data + "roi.png", roi_pano_img[...,::-1]) 

	# opencv built-in function for image stitching
	# stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
	# (status, stitched) = stitcher.stitch(images[0:10])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data",
		help="Choose what image you'd like to run on: one of listed above",
		type=str,
		choices=['shanghai', 'lab', 'river', 'indoor', 'road', 'hotel'],
		default='shanghai')
	parser.add_argument(
		"--candidate",
		help="Choose number of candidate",
		type=int,
		default=5)
	parser.add_argument(
		"--lowe_ratio",
		help="Choose lowe ratio used in feature matching",
		type=float,
		default=0.7)
	parser.add_argument(
		"--ransac_th",
		help="Choose ransac threshold value used in finding homography",
		type=int,
		default=4)
	parser.add_argument(
		"--roi_improve",
		help="Set true for those images with roi do not have good result",
		type=bool,
		default=False)
	arguments = parser.parse_args()
	main(arguments)


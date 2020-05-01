from final_part1 import point
from sklearn.neighbors import KDTree
import cv2
import numpy as np

def find_knn(img_pts_dict):
	# number of images
	n = len(img_pts_dict)
	# list, each element is a set of features in one image
	feat_list = []
	# number of features in one image
	n_feat = []
	# featrues to image index
	feat_to_img = {}

	for i in range(n):
		n_feat.append(len(img_pts_dict[i]))
		# extract features in each image
		points = img_pts_dict[i]
		# list, each element is a feature in one image
		feat_per_img = []
		for j in range(len(points)):
			feat_to_img[tuple(points[j].orb)] = i
			feat_per_img.append(points[j].orb.reshape(1,-1))

		feat_per_img = np.concatenate(feat_per_img, axis=0)
		feat_list.append(feat_per_img)

	# compute a big matrix for all features
	feature_set = np.concatenate(feat_list, axis=0)

	start = 0
	for i in range(n):
		# compute kd tree by all features except image i
		extract_feat = np.delete(feature_set, np.s_[start : start+n_feat[i]], axis=0)
		kd_tree = KDTree(extract_feat, leaf_size = 40)
		
		points = img_pts_dict[i]
		for j in range(n_feat[i]):
			knn_idx  = kd_tree.query(points[j].orb.reshape(1,-1), k=4, return_distance=False)
			knn_list = []
			for k in range(knn_idx.shape[1]):
				img_idx = feat_to_img[tuple(extract_feat[knn_idx[0, k]])]
				knn_list.append(img_idx)
			img_pts_dict[i][j].knn_list = knn_list
		start += n_feat[i]

	return img_pts_dict

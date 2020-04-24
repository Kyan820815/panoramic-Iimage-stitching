from final_part1 import point
from sklearn.neighbors import KDTree
import cv2
import numpy as np

def find_knn(dic):
	# number of images
	n = len(idc)
	# list, each element is a set of features in one image
	feat_list = []
	# list, each element is a feature in one image
	feat_per_img = []
	# number of features in one image
	n_feat = []

	# featrues to image index
	feat_to_img = {}

	for i in range(n):
		n_features.append(n_features, len(dic[i]))
		# extract features in each image
		points = dic[i]
		for j in range(len(points)):
			feat_to_img[points[j].orb] = i
			feat_per_img.append(feat_per_img, points[j].orb)

		feat_per_img = np.concatenate(feat_per_img, axis = 0)
		feat_list.append(feat_list, feat_per_img)

	# compute a big matrix for all features
	feature_set = np.concatenate(feat_list, axis = 0)

	start = 0
	for i in range(n):
		extract_feat = numpy.delete(feature_set, np.s_[start : start+n_features[i]], axis = 0)

		points = dic[i]
		for j in range(n_features[i]):
			features = np.concatenate((points[j].orb, extract_feat), axis=0)

			kd_tree 	 	= KDTree(features, leaf_size = 40)
			dist, min_idx 	= kd_tree.query(features[:1], k = 5)

			knn_list = []
			for k in range(1, len(min_idx)):
				knn_list.append(knn_list, feat_to_img[features[min_idx[k]]])

			dic[i][j].knn_list = knn_list

		start += n_features[i]


	return dic
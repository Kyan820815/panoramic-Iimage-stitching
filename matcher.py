from collections import Counter
import numpy as np 
import cv2

class matcher:
	def __init__(self, img_pts_dict, m_candidate, lowe_ratio, max_threshold):
		# config setting
		self.m_candidate   = m_candidate
		self.lowe_ratio    = lowe_ratio
		self.max_threshold = max_threshold
		# used for extract features and feature matching
		index_params  = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		self.surf  = cv2.xfeatures2d.SURF_create()
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)
		# compute candidate map
		self.candidate_map = self.candidate_match(img_pts_dict)

	def find_match(self, img_left, img_right):
		# obtain all possible matches between images
		matches, img_data = self.get_all_possible_match(img_left, img_right)
		# obtain valid matches from possible matches
		valid_matches = self.get_all_valid_match(matches)

		if len(valid_matches) > 4:
			pts_left  = img_data['left']['kp']
			pts_right = img_data['right']['kp']
			match_pts_left  = np.float32([pts_left[i].pt for (i, __) in valid_matches])
			match_pts_right = np.float32([pts_right[i].pt for (__, i) in valid_matches])
			# homography matrix from right to left
			H, s = self.compute_homography(match_pts_right, match_pts_left)
			return H
		
		return None

	def get_features(self, img):
		# make sure image is in format UINT-8
		img = np.uint8(img)
		kp, des = self.surf.detectAndCompute(img, None)

		return {'des':des, 'kp':kp}

	def get_all_possible_match(self, img_left, img_right):
		# obtain features
		data_left  = self.get_features(img_left)
		data_right = self.get_features(img_right)
		# compute matches given two feature sets
		all_matches = self.flann.knnMatch(data_right['des'], data_left['des'], k=2)

		return all_matches, {'left':data_left, 'right':data_right}

	def get_all_valid_match(self, all_matches):
		valid_matches = []
		for val in all_matches:
			# get valid matche using lowe concept
			if len(val) == 2 and val[0].distance < val[1].distance * self.lowe_ratio:
				valid_matches.append((val[0].trainIdx, val[0].queryIdx))

		return valid_matches

	def compute_homography(self, pointsA, pointsB):
		# compute homography matrix givin two point sets
		H, status = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, self.max_threshold)

		return (H, status)

	def candidate_match(self, img_pts_dict):
		candidate_map = {}
		for i in range(len(img_pts_dict)):
			allimg = []
			# candidate all image id
			for each in img_pts_dict[i]:
				allimg += each.knn_list
			# mapping from image id to count
			result = dict(Counter(allimg))
			# sorted reversely
			sorted_res = sorted(result.items(), key=lambda item: item[1], reverse=True)
			# m candidate matching images
			candidates = [each[0] for each in sorted_res[:self.m_candidate]]  
			candidate_map[i] = candidates

		return candidate_map

	def match_verification(self, inliers, matches):
		a = 8
		b = 0.3
		if inliers > a + b * matches:
			return True
		else:
			return False


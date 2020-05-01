from collections import Counter
import cv2
import numpy as np
from final_part1 import homography 

def candidate_match(img_pts_dict, m = 5):
	# m : number of candidates
	candidate_map = {}
	for i in range(len(img_pts_dict)):
		allimg = []  # candidate all image id
		for each in img_pts_dict[i]:
			allimg += each.knn_list
		result = dict(Counter(allimg))  # mapping from image id to count
		sorted_res = sorted(result.items(), key=lambda item: item[1], reverse=True)  # sorted reversely
		candidates = [each[0] for each in sorted_res[:m]]  # m candidate matching images
		candidate_map[i] = candidates
	return candidate_map


def get_allpossible_match(img_pts_dict, a, b):
	featuresA = []
	featuresB = []
	for each in img_pts_dict[a]:
		featuresA.append(each.orb)
	featuresA = np.array(featuresA)
	
	for each in img_pts_dict[b]:
		featuresB.append(np.array(each.orb))
	featuresB = np.array(featuresB)

	match_instance = cv2.DescriptorMatcher_create("BruteForce")    
	All_Matches = match_instance.knnMatch(featuresA, featuresB, 2)

	return All_Matches

	
def all_validmatches(AllMatches, owe_ratio):
		#to get all valid matches according to lowe concept.
		lowe_ratio = 0.75
		valid_matches = []

		for val in AllMatches:
			if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
				valid_matches.append((val[0].trainIdx, val[0].queryIdx))

		return valid_matches

def compute_homography(pointsA,pointsB,max_Threshold):
	#to compute homography using points in both images

	H, status = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
	return (H, status)

def match_verification(inliers, matches):
	a = 8
	b = 0.3
	if inliers > a + b * matches:
		return True
	else:
		return False

def find_img_pair(img_pts_dict, lowe_ratio, max_Threshold, m_candidate):
	# dictionary for each image
	img_homo_dict = {}

	# find candidates for each image
	candidate_map = candidate_match(img_pts_dict, m=m_candidate)

	# find image k and its candidates v (list) and check if this match is valid
	for k,v in candidate_map.items():
		# list of img k
		h_cells = []
		# for each candidate for image k
		for i in range(m_candidate):
			# compute the valid matches of image k and image v[i]
			pre_match = get_allpossible_match(img_pts_dict,k,v[i])
			per_match = all_validmatches(pre_match,lowe_ratio)
			if len(per_match) > 4:

				# pointsA: position of jth feature in image k.
				# j :feature index
				pointsA = np.float32([img_pts_dict[k][j].pos.pt for (_, j) in per_match])
				pointsB = np.float32([img_pts_dict[v[i]][j].pos.pt for (j, _) in per_match])

				h, status = compute_homography(pointsA, pointsB, max_Threshold)
				inliers = status.ravel().tolist()

				# if match_verification(len(inliers), len(per_match)) is True:

				# k and v[i] is valid match
				# h is homo from img k to img v[i]
				h_cell = homography(h, v[i])
				h_cells.append(h_cell)
							
		img_homo_dict[k] = h_cells

	return img_homo_dict

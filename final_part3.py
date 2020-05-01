from collections import Counter
import cv2
import numpy as np
from final_part1 import homography 

def candidate_match(dic, m = 5):
	# m : number of candidates
	candidate_map = {}
	for i in range(len(dic)):
		allimg = []  # candidate all image id
		for each in dic[i]:
			allimg += each.knn_list
		result = dict(Counter(allimg))  # mapping from image id to count
		sorted_res = sorted(result.items(), key=lambda item: item[1], reverse=True)  # sorted reversely
		candidates = [each[0] for each in sorted_res[:m]]  # m candidate matching images
		candidate_map[i] = candidates
	return candidate_map


def get_Allpossible_Match(dic, a, b):
	featuresA = []
	featuresB = []
	for each in dic[a]:
		featuresA.append(each.orb)
	featuresA = np.array(featuresA)
	
	for each in dic[b]:
		featuresB.append(np.array(each.orb))
	featuresB = np.array(featuresB)

	match_instance = cv2.DescriptorMatcher_create("BruteForce")    
	All_Matches = match_instance.knnMatch(featuresA, featuresB, 2)

	return All_Matches

	
def All_validmatches(AllMatches, owe_ratio):
		#to get all valid matches according to lowe concept.
		lowe_ratio = 0.75
		valid_matches = []

		for val in AllMatches:
			if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
				valid_matches.append((val[0].trainIdx, val[0].queryIdx))

		return valid_matches

def Compute_Homography(pointsA,pointsB,max_Threshold):
	#to compute homography using points in both images

	H, mask = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
	return (H, mask)

def match_verification(inliers, matches):
	a = 9
	b = 0.3
	if inliers > a + b * matches:
		return True
	else:
		return False

def matchKeypoints(dic, lowe_ratio, max_Threshold,m_candidate):

	valid_matches = [] # n*val; n = total matche pairs = image number*candidate per image; val: validmatches for each match pairs
	H = {}
	img_idx = {}

	candidate_map = candidate_match(dic, m=m_candidate)
	for k,v in candidate_map.items():
		I = []
		H[k] = {}
		for i in range(m_candidate):
			# compute the valid matches of image k and image v[i]
			pre_match = get_Allpossible_Match(dic,k,v[i])
			per_match = All_validmatches(pre_match,lowe_ratio)
			valid_matches.append(per_match)
			if len(per_match) > 4:
				# pointsA: position of jth feature in image k.
				# j :feature index
				pointsA = np.float32([dic[k][j].pos.pt for (_, j) in per_match])
				pointsB = np.float32([dic[v[i]][j].pos.pt for (j, _) in per_match])

				h, mask = Compute_Homography(pointsA, pointsB, max_Threshold)
				inliers = mask.ravel().tolist()
				if match_verification(len(inliers), len(per_match)) is True:
					H[k][i] = h
					I.append(v[i])
		if len(I) == 0:
			I = v

		img_idx[k] = I

	homo = homography(H, img_idx)
				
	return homo

		


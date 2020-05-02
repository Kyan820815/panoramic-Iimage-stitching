import cv2
import numpy as np
from matplotlib import pyplot as plt

def img_stitching(images, img_homo_dict):

	# start_idx = 0
	next_idx = 0
	stich_img = images[next_idx]
	# crt_img = images[0]
	# crt_homo = img_homo_dict[stich_img][0]

	stiched_list = [next_idx]

	# a = images[0]
	n = len(images)
	for i in range(15):
		
		pair_list = img_homo_dict[next_idx]
		print("=============== current image: ", next_idx)
		find = False
		for j in range(len(pair_list)):
			if pair_list[j].img_idx not in stiched_list:

				next_idx = pair_list[j].img_idx
				homo = pair_list[j].H
				img_pair = images[next_idx]
				stiched_list.append(next_idx)
				find = True
				break
		if find == False:
			return stich_img
		# print "Homography is : ", H
		# homo = compute_homography(stich_img, img_pair, max_Threshold=4.0)

		# dist = np.dot(homo, np.array([stich_img.shape[1], stich_img.shape[0], 1]));
		# dist = dist/dist[-1]
		# print "final ds=>", ds

		f1 = np.dot(homo, np.array([0,0,1]))
		f1 = f1/f1[-1]
		homo[0][-1] += abs(f1[0])
		homo[1][-1] += abs(f1[1])
		dist = np.dot(homo, np.array([stich_img.shape[1], stich_img.shape[0], 1]))
		offsety = abs(int(f1[1]))
		offsetx = abs(int(f1[0]))
		# dsize = (int(dist[0]) + offsetx, int(dist[1]) + offsety)
		dsize = (stich_img.shape[1] + offsetx , stich_img.shape[0] )
		# print "image dsize =>", dsize

		tmp = cv2.warpPerspective(stich_img, homo, dsize)
		# cv2.imshow("warped", tmp)
		# cv2.waitKey()
		print(dist[0])
		print(dist[1])
		print("x", offsetx)
		print("y", offsety)
		print("stich", stich_img.shape)
		print("tmp", tmp.shape)
		print("pair", img_pair.shape)

		plt.imshow(tmp, cmap='gray', vmin=0, vmax=255)
		plt.show()

		tmp[0:img_pair.shape[0] , (tmp.shape[1]-img_pair.shape[1]):tmp.shape[1]] = img_pair

		print(stich_img.shape)
		

		stich_img = tmp
		# crt_img = img_pair
		plt.imshow(stich_img, cmap='gray', vmin=0, vmax=255)
		plt.show()


	# self.leftImage = tmp

	plt.imshow(stich_img, cmap='gray', vmin=0, vmax=255)
	plt.show()


	return stich_img

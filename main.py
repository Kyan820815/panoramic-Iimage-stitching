from final_part1 import (load_data, get_ORB_feature)
from final_part2 import find_knn
from final_part3 import find_img_pair

def main():
	images = load_data("data/test1")
	img_pts_dict = get_ORB_feature(images)
	img_pts_dict = find_knn(img_pts_dict)
	img_homo_dict = find_img_pair(img_pts_dict, lowe_ratio=0.75, max_Threshold=4.0, m_candidate=5)

	# n = len(images)
	# for i in range(n):
	# 	print("====== imgidx: ", i)
	# 	homo_list = img_homo_dict[i]
	# 	for j in range(len(homo_list)):
	# 		print(homo_list[j].img_idx)

if __name__ == '__main__':
	main()


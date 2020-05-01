from final_part1 import (load_data, get_ORB_feature)
from final_part2 import find_knn
from final_part3 import matchKeypoints

def main():
	images = load_data("data/test")
	dic = get_ORB_feature(images)
	dic = find_knn(dic)
	homo = matchKeypoints(dic, lowe_ratio=0.75, max_Threshold=4.0, m_candidate=5)

if __name__ == '__main__':
	main()

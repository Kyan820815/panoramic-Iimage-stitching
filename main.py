from final_part1 import (load_data, get_ORB_feature)
from final_part2 import find_knn

def main():
	images = load_data("data")
	dic = get_ORB_feature(images)
	dif = find_knn(dic)

if __name__ == '__main__':
	main()

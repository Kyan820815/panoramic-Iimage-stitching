import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree


class point:
    def __init__(self, pos, orb, knn_list):
        self.pos = pos # position information of key point
        self.orb = orb # orb feature
        self.knn_list = knn_list


class preprocesser:
    def __init__(self, path, draw=False):
        self.img_pts_dict = {}
        self.load_data(path)
        self.get_ORB_feature(draw)
        self.find_knn()

    def load_data(self, path):
        cnt   = 0
        files = os.listdir(path)
        l = len(files)
        self.images = [0] * l
        for file in files:
            if file == ".DS_Store":
                continue
            idx = int(file[-6:-4]) - 1 
            self.images[idx] = cv2.imread(path + '/' + file)
            if len(self.images[idx].shape) == 3:
                # change to opencv reading image format
                self.images[idx] = self.images[idx][...,::-1]
                # resize image if it is too big
            if self.images[idx].shape[0] > 1000:
                self.images[idx] = cv2.resize(self.images[idx], (2016, 1512))
                # images[idx] = cv2.resize(images[idx], (1008, 756))
            cnt += 1

        self.images = self.images[:cnt]
        self.images = np.array(self.images)


    def get_ORB_feature(self, draw):
        orb = cv2.ORB_create()
        for i in range(len(self.images)):
            # get orb features of each image
            kp, des = orb.detectAndCompute(self.images[i], None)

            if draw == True:
                img = cv2.drawKeypoints(self.images[i], kp, np.array([]))
                plt.imshow(img)
                plt.show()

            # store each point's information in one list and map to the image
            pts = []
            for j in range(len(kp)):
                pt = point(kp[j], des[j], [])
                pts.append(pt)
            self.img_pts_dict[i] = pts

    def find_knn(self):
        # number of images
        n = len(self.img_pts_dict)
        # list, each element is a set of features in one image
        feat_list = []
        # number of features in one image
        n_feat = []
        # featrues to image index
        feat_to_img = {}

        for i in range(n):
            n_feat.append(len(self.img_pts_dict[i]))
            # extract features in each image
            points = self.img_pts_dict[i]
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
            print("------- find knn features for image",i)
            # compute kd tree by all features except image i
            extract_feat = np.delete(feature_set, np.s_[start : start+n_feat[i]], axis=0)
            kd_tree = KDTree(extract_feat, leaf_size = 40)
            
            points = self.img_pts_dict[i]
            for j in range(n_feat[i]):
                knn_idx  = kd_tree.query(points[j].orb.reshape(1,-1), k=4, return_distance=False)
                knn_list = []
                for k in range(knn_idx.shape[1]):
                    img_idx = feat_to_img[tuple(extract_feat[knn_idx[0, k]])]
                    knn_list.append(img_idx)
                self.img_pts_dict[i][j].knn_list = knn_list
            start += n_feat[i]


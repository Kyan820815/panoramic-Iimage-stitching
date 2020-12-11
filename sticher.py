from matplotlib import pyplot as plt
from matcher import matcher
import numpy as np
import imutils
import cv2
import operator
import math
from PIL import Image
from PIL import ImageDraw


class sticher:
    """
    Class for stiching images based on homography matrix and their relations
    We stitch images based on the direction of the whole picture
    """
    def __init__(self, matcher, roi_improve=False):
        self.matcher = matcher
        self.roi_improve = roi_improve

    def img_stitch_left(self, image_left, image_right, homo):
        # compute shift shift x and y from img left to img right
        shift  = np.dot(homo, np.array([0,0,1]))
        # make projection homogeneous index = 1
        shift  = shift / shift[-1]
        shifty = int(abs(shift[1]))
        shiftx = int(abs(shift[0]))
        # change homography matrix that fit left -> right
        homo = np.linalg.inv(homo)
        # update translation x, y
        homo[0, 2] += abs(shift[0])
        homo[1, 2] += abs(shift[1])
        # find projection axis from left img to right img
        end_points = np.dot(homo, np.array([image_left.shape[1], image_left.shape[0], 1]))
        # make projection homogeneous index = 1
        end_points = end_points / end_points[-1]

        # compute stiched image size
        img_size = (int(end_points[0]), int(end_points[1]))
        # warp left image to fit right image
        warp_image_left = cv2.warpPerspective(image_left, homo, img_size)
        # re-define stitch image size
        max_row = max(image_right.shape[0]+shifty, warp_image_left.shape[0])
        max_col = max(image_right.shape[1]+shiftx, warp_image_left.shape[1])
        # merge left and right image
        if len(warp_image_left.shape) == 2:
            stitch_img = np.zeros((max_row, max_col))
        else:
            stitch_img = np.zeros((max_row, max_col, 3))
        stitch_img[0:warp_image_left.shape[0], 0:warp_image_left.shape[1]] = warp_image_left
        stitch_img[shifty:image_right.shape[0]+shifty, shiftx:image_right.shape[1]+shiftx] = image_right

        return stitch_img

    def img_stitch_right(self, image_left, image_right, homo):
        # find projection axis from right img to left img
        end_points = np.dot(homo, np.array([image_right.shape[1], image_right.shape[0], 1]))
        # make projection homogeneous index = 1
        end_points = end_points / end_points[-1]
        img_size = (int(end_points[0]), int(end_points[1]))
        warp_image_right = cv2.warpPerspective(image_right, homo, img_size)

        # re-define stitch image size
        max_row = max(image_left.shape[0], warp_image_right.shape[0])
        max_col = max(image_left.shape[1], warp_image_right.shape[1])
        for i in range(max_col - warp_image_right.shape[1]):
            warp_image_right = np.column_stack((warp_image_right, np.zeros((warp_image_right.shape[0], 1, 3))))
        for i in range(max_row - warp_image_right.shape[0]):
            warp_image_right = np.row_stack((warp_image_right, np.zeros((1, max_col, 3))))

        # since there might some back area in image left but not int image right
        # we copy those area from image right to image left to remove these back area in image left
        extract_right = warp_image_right[0:image_left.shape[0], 0:image_left.shape[1]]
        extract_right[image_left > 0] = 0
        image_left += np.uint8(extract_right)
        # merge left and right image
        warp_image_right[0:image_left.shape[0], 0:image_left.shape[1]] = image_left
        
        return warp_image_right

    def img_pano(self, images, cur_idx=0):
        print("------- apply image stiching to panorama")
        stitch_img = images[cur_idx]
        stiched_list = [cur_idx]

        n = len(images)
        for i in range(n-1):
            pair_list = self.matcher.candidate_map[cur_idx]
            find_next_idx = False
            for j in range(len(pair_list)):
                if pair_list[j] > n or pair_list[j] < cur_idx:
                    continue
                if pair_list[j] not in stiched_list:
                    # update index to image needed to be stitched
                    cur_idx  = pair_list[j]
                    img_pair = images[cur_idx]
                    homo = self.matcher.find_match(stitch_img, img_pair)
                    # not valid match pair, skip
                    if homo is None:
                        continue
                    # record image that has already been stitched
                    stiched_list.append(cur_idx)
                    find_next_idx = True
                    break
            # reach last image
            if find_next_idx == False:
                return stitch_img

            if i < int(n/2):
                # stich from left to right
                stitch_img = self.img_stitch_left(stitch_img, img_pair, homo)
            else:
                # stich from right to left
                stitch_img = self.img_stitch_right(stitch_img, img_pair, homo)
            # plt.imshow((stitch_img).astype(np.uint8))
            # plt.show()

        pano_image = stitch_img
        # use roi toward pano image
        print("------- apply roi on panorama image")
        roi_pano_image = self.img_roi(pano_image)

        return pano_image, roi_pano_image

    def img_roi(self, image):
        # compute threshold mask based on image pixel value
        # using boarder can make us find roi easily
        image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        r_ch = image[:,:,0].copy()
        g_ch = image[:,:,1].copy()
        b_ch = image[:,:,2].copy()
        r_ch[r_ch > 0] = 1
        g_ch[g_ch > 0] = 1
        b_ch[b_ch > 0] = 1
        image_th = r_ch + g_ch + b_ch
        image_th[image_th > 0] = 255
        image_th = np.uint8(image_th)

        # find min boarder from down to top in the midian of col
        # for roi improvement, usd in heavily twisted image
        if self.roi_improve == True:
            midean = int(image_th.shape[1] / 2)
            for i in range(image_th.shape[0]-1, 0, -1):
                if image_th[i, midean] == 255:
                    min_boarder = i
                    break
            image_th = image_th[:i,:]
            image    = image[:i,:]
            image_th = cv2.copyMakeBorder(image_th, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
            image    = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

        # find countours of mask image
        contours = cv2.findContours(image_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        # find largest contour based on mask and cv contourArea
        c = max(contours, key=cv2.contourArea)
        # obtain roi image
        roi = np.zeros(image_th.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(roi, (x, y), (x+w, y+h), 255, -1)
        
        # plt.imshow((image_th).astype(np.uint8))
        # plt.show()

        # find minimum roi that do not contain any black pixel
        min_roi = roi.copy()
        sub     = roi.copy()
        while cv2.countNonZero(sub) > 0:
            # update roi
            min_roi = cv2.erode(min_roi, None)
            sub = cv2.subtract(min_roi, image_th)

        # find countours of best roi image
        contours = cv2.findContours(min_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        # find largest contour based on roi and cv contourArea
        best_contours = max(contours, key=cv2.contourArea)
        (x, y, w, h)  = cv2.boundingRect(best_contours)
        # roi the image
        image = image[y:y+h, x:x+w]
        
        return image

    
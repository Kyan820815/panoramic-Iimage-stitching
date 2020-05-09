# CSCI1430-Final-Project
Automatic Panoramic Image Stitching  

In this project, we build panorama image stitching with unordered data.

The command is: 
```
	main.py [-h] [--data {shanghai,lab,river,indoor,road,hotel}]
               	 [--candidate CANDIDATE] [--lowe_ratio LOWE_RATIO]
                 [--ransac_th RANSAC_TH] [--roi_improve ROI_IMPROVE]
```

optional arguments:
```
  -h, --help            show this help message and exit
  --data {shanghai,lab,river,indoor,road,hotel}
                        Choose what image you'd like to run on: one of listed
                        above
  --candidate CANDIDATE
                        Choose number of candidate
  --lowe_ratio LOWE_RATIO
                        Choose lowe ratio used in feature matching
  --ransac_th RANSAC_TH
                        Choose ransac threshold value used in finding
                        homography
  --roi_improve ROI_IMPROVE
                        Set true for those images with roi do not have good
                        result
```

* The data set is on: https://www.dropbox.com/sh/kui1xs38o15xbaw/AACxJ7g6ci0qz_nG0rjujIcMa?dl=0

* Please create data folder in the code directory and result folder in the data folder


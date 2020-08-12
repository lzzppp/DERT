# The Market-1501 dataset was collected on the campus of Tsinghua University, shot in the summer, and constructed and made public in 2015. It includes 1501 
# pedestrians and 32,668 detected pedestrian rectangles captured by 6 cameras (including 5 high-definition cameras and 1 low-definition camera). Each pedestrian 
# is captured by at least 2 cameras, and there may be multiple images in one camera. The training set has 751 people, containing 12,936 images, and each person 
# has an average of 17.2 training data; the test set has 750 people, containing 19,732 images, and each person has an average of 26.3 test data.

# Directory Structure
# Market-1501
# 　　├── bounding_box_test
# 　　　　　　　├── 0000_c1s1_000151_01.jpg
# 　　　　　　　├── 0000_c1s1_000376_03.jpg
# 　　　　　　　├── 0000_c1s1_001051_02.jpg
# 　　├── bounding_box_train
# 　　　　　　　├── 0002_c1s1_000451_03.jpg
#  　 　　　　　├── 0002_c1s1_000551_01.jpg
# 　　　　　　　├── 0002_c1s1_000801_01.jpg
# 　　├── gt_bbox
# 　　　　　　　├── 0001_c1s1_001051_00.jpg
# 　　　　　　　├── 0001_c1s1_009376_00.jpg
# 　　　　　　　├── 0001_c2s1_001976_00.jpg
# 　　├── gt_query
# 　　　　　　　├── 0001_c1s1_001051_00_good.mat
# 　　　　　　　├── 0001_c1s1_001051_00_junk.mat
# 　　├── query
# 　　　　　　　├── 0001_c1s1_001051_00.jpg
# 　　　　　　　├── 0001_c2s1_000301_00.jpg
# 　　　　　　　├── 0001_c3s1_000551_00.jpg
# 　　└── readme.txt

# Catalog Introduction
# The pictures in the bounding_box_train folder and the bounding_box_test folder are source data for generating matching and non-matching.

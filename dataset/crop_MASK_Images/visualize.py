import os
from PIL import Image
import cv2
import glob

images = glob.glob("./test_max_infer_0.005/*.png")

for img in images:
    img_name = img.split("/")[-1]
    teeth_img = cv2.imread("{}".format(img))
    cv2.imwrite("./test_max_infer_0.005_vis/{}".format(img_name), teeth_img*15)
    exit()
    
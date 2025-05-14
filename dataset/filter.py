import os
import glob
from PIL import Image
import cv2


# mask_image = glob.glob("./MASK_Images/test/*.png")

# for img in mask_image:
#     img_name = img.split("/")[-1]
#     image = Image.open("./Infer_SegMap3/{}.png".format(img_name))
#     print(image)
#     exit()

crop_MASK_Images = glob.glob("./crop_MASK_Images/test_man_infer_0.005/*.png")

for i in crop_MASK_Images:
    img_name = i.split("/")[-1].split(".")[0].split("_")[0]
    img_num = i.split("/")[-1].split(".")[0].split("_")[1]
    if img_name == "705450000":
        print(img_name)
        img = cv2.imread("{}".format(i))
        cv2.imwrite("./vis/{}_{}.png".format(img_name, img_num), img *25)
    else:
        pass
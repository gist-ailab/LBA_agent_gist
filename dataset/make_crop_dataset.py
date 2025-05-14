import os
import numpy 
import glob
import json
import cv2
from tqdm import tqdm
from PIL import Image

#img_list = glob.glob("/data_3/JSLEE/Wisdom_classification_v5/dataset/Infer_SegMap3/*.png")
img_list = glob.glob("/data2/JSLEE/Third_molar/classification/dataset/MASK_Images/test_0.005/*.png")


for img in tqdm(img_list):

    img_name = img.split("/")[-1].split(".")[0] 

    with open("./Annotations/{}.jpg.png.json".format(img_name)) as json_file:
        #img_array = cv2.imread("/data_3/JSLEE/Wisdom_classification_v5/dataset/Infer_SegMap3/{}.jpg.png".format(img_name))
        img_array = cv2.imread("/data2/JSLEE/Third_molar/classification/dataset/MASK_Images/test_0.005/{}.jpg.png".format(img_name))

        json_data = json.load(json_file)
        json_object = json_data["objects"]

        x_list_18 = []
        y_list_18 = []
        x_list_28 = []
        y_list_28 = []
        x_list_38 = []
        y_list_38 = []
        x_list_48 = []
        y_list_48 = []
        for k in range(len(json_object)):
            json_classTitle = json_object[k]["classTitle"]
            # #18에 point list에서 x,y에 대한 min max
            if json_classTitle == "#18":
                json_exterior = json_object[k]["points"]["exterior"]
                for f in range(len(json_exterior)):
                    x_list_18.append(json_exterior[f][0])
                    y_list_18.append(json_exterior[f][1])
                x_max_18 = max(x_list_18)
                y_max_18 = max(y_list_18)
                x_min_18 = min(x_list_18)
                y_min_18 = min(y_list_18)
                img_array = cv2.rectangle(img_array, (min(x_list_18), min(y_list_18)), (max(x_list_18), max(y_list_18)),(0,0,255), 3)
                #image = Image.open("/data_3/JSLEE/Wisdom_classification_v5/dataset/Infer_SegMap3/{}.jpg.png".format(img_name))
                image = Image.open("/data2/JSLEE/Third_molar/classification/dataset/MASK_Images/test_0.005/{}.jpg.png".format(img_name))
                width = x_max_18 - x_min_18 
                height =  y_max_18 - y_min_18
                width_18 = 700 - width
                height_18 = 700 - height 

                image_18 = image.crop((x_min_18 - int(width_18/2), y_min_18 - int(height_18/2), x_max_18 + int(width_18/2), y_max_18 + int(height_18/2)))

                #image_18.save("./Infer_SegMap3_crop_/{}_18.png".format(img_name))
                image_18.save("/data2/JSLEE/Third_molar/classification/dataset/crop_MASK_Images/test_max_infer_0.005/{}_18.png".format(img_name))
                


            elif json_classTitle == "#28":
                json_exterior = json_object[k]["points"]["exterior"]
                for f in range(len(json_exterior)):
            
                    x_list_28.append(json_exterior[f][0])
                    y_list_28.append(json_exterior[f][1])
                x_max_28 = max(x_list_28)
                y_max_28 = max(y_list_28)
                x_min_28 = min(x_list_28)
                y_min_28 = min(y_list_28)
                img_array = cv2.rectangle(img_array, (min(x_list_28), min(y_list_28)), (max(x_list_28), max(y_list_28)),(0,0,255), 3)
                #image = Image.open("/data_3/JSLEE/Wisdom_classification_v5/dataset/Infer_SegMap3/{}.jpg.png".format(img_name))
                image = Image.open("/data2/JSLEE/Third_molar/classification/dataset/MASK_Images/test_0.005/{}.jpg.png".format(img_name))
    
                width = x_max_28 - x_min_28
                height = y_max_28 - y_min_28
                width_28 = 700 - width
                height_28 = 700 - height 

                image_28 = image.crop((x_min_28 - int(width_28/2), y_min_28 - int(height_28/2), x_max_28 + int(width_28/2), y_max_28 + int(height_28/2)))
                #image_28.save("./Infer_SegMap3_crop_/{}_28.png".format(img_name))
                image_28.save("/data2/JSLEE/Third_molar/classification/dataset/crop_MASK_Images/test_max_infer_0.005/{}_28.png".format(img_name))


            elif json_classTitle == "#38":
                json_exterior = json_object[k]["points"]["exterior"]
                for f in range(len(json_exterior)):
        
                    x_list_38.append(json_exterior[f][0])
                    y_list_38.append(json_exterior[f][1])
                x_max_38 = max(x_list_38)
                y_max_38 = max(y_list_38)
                x_min_38 = min(x_list_38)
                y_min_38 = min(y_list_38)
                img_array = cv2.rectangle(img_array, (min(x_list_38), min(y_list_38)), (max(x_list_38), max(y_list_38)),(0,0,255), 3)
                #image = Image.open("/data_3/JSLEE/Wisdom_classification_v5/dataset/Infer_SegMap3/{}.jpg.png".format(img_name))
                image = Image.open("/data2/JSLEE/Third_molar/classification/dataset/MASK_Images/test_0.005/{}.jpg.png".format(img_name))
              
                width = x_max_38 - x_min_38
                height =  y_max_38 - y_min_38
                width_38 = 700 - width
                height_38 = 700 - height 
                image_38 = image.crop((x_min_38 - int(width_38/2), y_min_38 - int(height_38/2), x_max_38 + int(width_38/2), y_max_38 + (height_38/2)))
                #image_38.save("./Infer_SegMap3_crop_/{}_38.png".format(img_name))
                image_38.save("/data2/JSLEE/Third_molar/classification/dataset/crop_MASK_Images/test_man_infer_0.005/{}_38.png".format(img_name))


            elif json_classTitle == "#48":
                json_exterior = json_object[k]["points"]["exterior"]
                for f in range(len(json_exterior)):
        
                    x_list_48.append(json_exterior[f][0])
                    y_list_48.append(json_exterior[f][1])
                x_max_48 = max(x_list_48)
                y_max_48 = max(y_list_48)
                x_min_48 = min(x_list_48)
                y_min_48 = min(y_list_48)
                img_array = cv2.rectangle(img_array, (min(x_list_48), min(y_list_48)), (max(x_list_48), max(y_list_48)),(0,0,255), 3)
                #image = Image.open("/data_3/JSLEE/Wisdom_classification_v5/dataset/Infer_SegMap3/{}.jpg.png".format(img_name))
                image = Image.open("/data2/JSLEE/Third_molar/classification/dataset/MASK_Images/test_0.005/{}.jpg.png".format(img_name))

                width = x_max_48 - x_min_48
                height =  y_max_48 - y_min_48
                width_48 = 700 - width
                height_48 = 700 - height 
                image_48 = image.crop((x_min_48 - int(width_48/2), y_min_48 - int(height_48/2), x_max_48 + int(width_48/2), y_max_48 + int(height_48/2)))
                #image_48.save("./Infer_SegMap3_crop_/{}_48.png".format(img_name))
                image_48.save("/data2/JSLEE/Third_molar/classification/dataset/crop_MASK_Images/test_man_infer_0.005/{}_48.png".format(img_name))

            
            else :
                pass
            
           

            
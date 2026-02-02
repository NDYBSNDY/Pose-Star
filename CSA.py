import cv2
import numpy as np
import os

def getfiles(path):
    filenames = os.listdir(path)
    filepaths = []
    for item in filenames:
        file = path + item
        filepaths.append(file)
    return filepaths, filenames

def get_imgatt(path_img, path_att, path_save):
    img = cv2.imread(path_img, 1)    
    att = cv2.imread(path_att,1)    
    w = cv2.applyColorMap(att, 2)    
    x = img * 0.5 + w * 0.5   
    x = x.astype(np.uint8)
    cv2.imwrite(path_save,x)

def get_att(path_att, path_save):
    att = cv2.imread(path_att,1)   
    w = cv2.applyColorMap(att, 2)    
    w_ = w.astype(np.uint8)
    cv2.imwrite(path_save,w_)

def get_mask(path_att, path_save):
    custom_colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    custom_colormap[0:100] = [0, 0, 0]  
    custom_colormap[100:256] = [255, 255, 255]  
    gray_image = cv2.imread(path_att, cv2.IMREAD_GRAYSCALE)

    color_image = cv2.applyColorMap(gray_image, custom_colormap)
    color_image = color_image.astype(np.uint8)
    cv2.imwrite(path_save, color_image)



get_imgatt(img_path, path_att, path_save_imgatt)
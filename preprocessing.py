import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image

def hair_removal(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,thresh2,1,cv2.INPAINT_TELEA)
    return dst

def resize_img(img):
    if img.shape[0] == 224 and img.shape[1] == 224:
        return img
    else:
        img_resize = cv2.resize(img, (224,224))
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        return img_resize

def classifier(img):
    model = tf.keras.models.load_model("simpler.h5", custom_objects=None, compile=True, options=None)
    im_pil = Image.fromarray(img)
    im_np = np.asarray(im_pil)
    img_reshape = im_np[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction
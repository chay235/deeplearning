import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import img_to_array

def set_background(image_file):...

def classify(image, model, class_names):
        #convert image to 224,224
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    plt.figure(figsize=(10, 10))
    plt.subplot(1,3,1)
    plt.imshow(img, cmap="gray")
    plt.axis('off') 
    plt.show
    i = img_to_array(img)
    input_arr = np.array([i])
    predict_x=model.predict(input_arr)
    class_x=np.argmax(predict_x)
    pred_accuracy = predict_x.max()
    print(predict_x)
    print(class_x)
    if(class_x == 1):
        print("The prediction on MRI IMAGE is GLIOMA TUMOR")
        class_label = "Glioma Tumor"
    elif(class_x == 2):
        print("The prediction on MRI IMAGE is MENINGIOMA TUMOR")
        class_label = "Meningioma Tumor"
    elif(class_x == 3):
        print("The prediction on MRI IMAGE is PITUITARY TUMOR")
        class_label = "Pituitary Tumor"
    else:
        print("The MRI Image is Of HEALTHY BRAIN")
        class_label = "Healthy brain"
    return class_label, pred_accuracy

import streamlit as st
import tensorflow as tf
import keras
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify
from streamlit_modal import Modal

#title part of UI
st.title('Brain Tumor Analyser and Classifier')
#header part
st.header('Please upload MRI image of brain')
#upload the file
file = st.file_uploader('',type=['jpeg','jpg'])
vggmodelpath='C:/Users/kvsk_/Downloads/vggmodel_web.h5'
efficientnetmodelpath='C:/Users/kvsk_/Downloads/efficientNetmodel_New.h5'
sel_model = st.radio(
    "Select model to classify",
    ["VGG16", "EfficientNet" ],
    index=0,
)
print(sel_model)
st.write("You selected:", sel_model)
# display image
if (sel_model == "VGG16"):
    modelpath = vggmodelpath
else:
    modelpath = efficientnetmodelpath


plots_container=st.container()
with plots_container:
    st.write("Plots :")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col2:
        modal_acc_vgg = Modal(key="Acc_plot",title="Plots")
        open_acc_vggmodal = st.button("VGG Accuracy")
        if open_acc_vggmodal:
            modal_acc_vgg.open()

    
    with col3:
        vgg_modal_loss = Modal(key="Loss_plot_vgg",title="Plots")
        open_loss_vgg_modal = st.button("VGG Loss")
        if open_loss_vgg_modal:
            vgg_modal_loss.open()

   

    with col4:
        modal_acc_eff = Modal(key="Acc_plots",title="Plots")
        open_acc_effmodal = st.button("EffNet Accuracy")
        if open_acc_effmodal:
            modal_acc_eff.open()

    with col5:
        eff_modal_loss = Modal(key="Loss_plot_eff",title="Plots")
        open_loss_eff_modal = st.button("EffNet Loss")
        if open_loss_eff_modal:
            eff_modal_loss.open()



if modal_acc_vgg.is_open():
        with modal_acc_vgg.container():
            vgg_acc_image = Image.open('C:/Users/kvsk_/Downloads/Accuracy_plot_vgg.png')
            st.image(vgg_acc_image,caption='Accuracy plot', width=500)


if vgg_modal_loss.is_open():
        with vgg_modal_loss.container():
            vgg_loss_image = Image.open('C:/Users/kvsk_/Downloads/loss_plot_vgg.png')
            st.image(vgg_loss_image,caption='Loss plot',width=500)

if eff_modal_loss.is_open():
    with eff_modal_loss.container():
        eff_loss_image = Image.open('C:/Users/kvsk_/Downloads/loss_plot_effc.png')
        st.image(eff_loss_image,caption='Loss plot',width=500)


if modal_acc_eff.is_open():
    with modal_acc_eff.container():
        eff_acc_image = Image.open('C:/Users/kvsk_/Downloads/Accuracy_plot_effc.png')
        st.image(eff_acc_image,caption='Accuracy plot', width=500)

model = tf.keras.models.load_model(modelpath,compile=False)

with open('C:/Users/kvsk_/Downloads/labels/labels.txt','r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
print(class_names)

if file is not None :
    print("inside classification")
    image = Image.open(file).convert('RGB')
    st.image(image,width = 500)
    #classify image
    class_name, conf_score = classify(image,model,class_names)

    #write classification
    if (class_name == "Healthy brain"):
        st.write("No tumor found, it is a","{}".format(class_name))
        st.write("These results are not accurate, contact Medical expert.")
    else:
        st.write("The class of tumor predicted is","{}".format(class_name))
        st.write("These results are not accurate, contact Medical expert.")

    st.write("Accuracy score: {}%".format(int(conf_score * 1000) / 10))

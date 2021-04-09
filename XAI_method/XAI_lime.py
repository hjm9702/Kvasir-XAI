#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, pickle, csv
import cv2

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, auc, roc_curve

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as VGG19_preprocess
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as InceptionV3_preprocess
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as ResNet50V2_preprocess
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as Xception_preprocess

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.cm as cm



from skimage.segmentation import mark_boundaries
from lime import lime_image
import lime


class XAI_lime():
    
    def __init__(self, model, model_name, preprocess_func, img_dim = 384):
        self.model = model
        self.model_name = model_name
        self.preprocess_func = preprocess_func
        self.img_dim = img_dim
        
    
    def make_lime_heatmap(self, img):
        #Get explanation from lime
        
        img_preprocessed = self.preprocess_func(img)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img_preprocessed.astype('double'), self.model.predict, hide_color=0, num_samples=1000)
        #_, mask = explanation.get_image_and_mask(7, positive_only = True)
        
        dict_heatmap = dict(explanation.local_exp[7])
        #label_index is 7 for polyp class
        
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        

        return heatmap
    
    
    def lime_single_auroc(self, X_tst, Y_tst):
        #Calculate auroc for each image and mask
        img = X_tst.copy()
        
        heatmap = self.make_lime_heatmap(img)
        heatmap_copy = heatmap.copy()
        fpr, tpr, _ = roc_curve(Y_tst.ravel(), heatmap_copy.ravel())
        roc_auc = auc(fpr, tpr)
        
        return heatmap, roc_auc
    
    def lime_total_auroc(self, X_tst, Y_tst):
        #Calculate auroc for whole dataset
        heatmap_list = []
        auroc_list = []
        
        for i in range(len(X_tst)):
            heatmap, roc_auc = self.lime_single_auroc(X_tst[i], Y_tst[i])
            
            heatmap_list.append(heatmap)
            auroc_list.append(roc_auc)
        
        #with open('./kvasir_lime_heatmaps_'+str(self.model_name)+'.pickle', 'wb') as f:
        #    pickle.dump(heatmap_list, f)
        
        
        
        return sum(auroc_list) / len(auroc_list)
    
    
    def lime_single_visualize(self, X_tst, Y_tst, alpha):
        img = X_tst.copy()
        
        heatmap = self.make_lime_heatmap(img)
        heatmap_processed = np.float32(heatmap)
        heatmap_processed = cv2.applyColorMap(((heatmap_processed- heatmap_processed.min())/
                                               (heatmap_processed.max() - heatmap_processed.min())*255).astype('uint8'), cv2.COLORMAP_JET)
        heatmap_processed = cv2.cvtColor(heatmap_processed, cv2.COLOR_BGR2RGB)
 
        
        
        superimposed_img = (heatmap_processed * alpha + img * (1 - alpha)).astype('uint8')
        
        fig, axs = plt.subplots(1, 4, figsize=(3*12, 1*12))
        axs[0].imshow(heatmap_processed)
        axs[0].set_title('Lime heatmap', fontsize = 30)
        axs[1].imshow(img)
        axs[1].set_title('Input Image', fontsize = 30)
        axs[2].imshow(superimposed_img)
        axs[2].set_title('Superimposed Image', fontsize = 30)
        axs[3].imshow(Y_tst)
        axs[3].set_title('Mask', fontsize = 30)

        for ax in axs.flat:
            ax.label_outer()
 
    
    '''
    def lime_single_visualize_faster(self, X_tst, Y_tst, alpha, idx):
        
        try:
            with open('./kvasir_lime_heatmaps_'+str(self.model_name)+'.pickle', 'rb') as f:
                total_heatmaps = pickle.load(f)
            
            img = X_tst.copy()    
            
            heatmap = total_heatmaps[idx]
            heatmap_processed = np.float32(heatmap)
            heatmap_processed = cv2.applyColorMap(((heatmap_processed- heatmap_processed.min())/
                                               (heatmap_processed.max() - heatmap_processed.min())*255).astype('uint8'), cv2.COLORMAP_JET)
            heatmap_processed = cv2.cvtColor(heatmap_processed, cv2.COLOR_BGR2RGB)

            superimposed_img = (heatmap_processed * alpha + img * (1 - alpha)).astype('uint8')

            fig, axs = plt.subplots(1, 4, figsize=(3*12, 1*12))
            axs[0].imshow(heatmap_processed)
            axs[0].set_title('Lime heatmap', fontsize = 30)
            axs[1].imshow(img)
            axs[1].set_title('Input Image', fontsize = 30)
            axs[2].imshow(superimposed_img)
            axs[2].set_title('Superimposed Image', fontsize = 30)
            axs[3].imshow(Y_tst)
            axs[3].set_title('Mask', fontsize = 30)

            for ax in axs.flat:
                ax.label_outer()
            
        
        
        except:
            raise KeyError('lime_total_auroc method should be done!')

    '''

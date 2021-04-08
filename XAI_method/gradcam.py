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


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # memory increase is needed before starting program
        print(e)


class GradCam():
    def __init__(self, model, last_conv_layer_name, img_dim = 384, pred_index=None):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.img_dim = img_dim
        self.pred_index = pred_index

    # pred_index = 7 for polyp class
    def make_gradcam_heatmap(self, img_array):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(np.expand_dims(img_array, axis=0).astype('float32'))  # uint8 -> float32
            if self.pred_index is None:
                pred_index = tf.argmax(preds[0])
            else:
                pred_index = self.pred_index
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    # calculate gradcam of each image and mask
    # X_tst, Y_tst are single image, maks (img_w, img_h, color_channel)
    def gradcam_single_auroc(self, X_tst, Y_tst):
        heatmap = self.make_gradcam_heatmap(X_tst)
        heatmap = cv2.resize(heatmap, (self.img_dim, self.img_dim))
        fpr, tpr, _ = roc_curve(Y_tst.ravel(), heatmap.ravel())
        roc_auc = auc(fpr, tpr)
        return roc_auc

    # calculate whole dataset auroc
    # X_tst, Y_tst are whole dataset, consist of 4 dimension (instances #, img_w, img_h, color_channel)
    def gradcam_total_auroc(self, X_tst, Y_tst):
        auroc_list = []
        for i in range(len(X_tst)):
            auroc_list.append(self.gradcam_single_auroc(X_tst[i], Y_tst[i]))
        return sum(auroc_list) / len(auroc_list)

    # visualize heatmap on the image
    # the img_array must not be preprocessed
    def gradcam_visualize(self, img_array_org, Y_tst, alpha):
        img_array = img_array_org.copy()
        heatmap = self.make_gradcam_heatmap(img_array)
        heatmap = cv2.resize(heatmap, (self.img_dim, self.img_dim))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        decoded_image = self.deprocess_img(img_array)
        superimposed_img = (heatmap * alpha + decoded_image * (1 - alpha)).astype('uint8')

        fig, axs = plt.subplots(1, 4, figsize=(3*12, 1*12))
        axs[0].imshow(heatmap)
        axs[0].set_title('GradCam heatmap', fontsize = 30)
        axs[1].imshow(decoded_image)
        axs[1].set_title('Input Image', fontsize = 30)
        axs[2].imshow(superimposed_img)
        axs[2].set_title('Superimposed Image', fontsize = 30)
        axs[3].imshow(Y_tst)
        axs[3].set_title('Mask', fontsize = 30)

        for ax in axs.flat:
            ax.label_outer()

        return heatmap, decoded_image, superimposed_img

    def deprocess_img(self, img_array):

        # Model : VGG19
        if self.last_conv_layer_name == 'block5_conv4':
            img_array[:, :, 0] += 103.939
            img_array[:, :, 1] += 116.779
            img_array[:, :, 2] += 123.68
            img_array = img_array[:, :, ::-1]
            img_array = np.clip(img_array, 0, 255).astype('uint8')

        # Model : InceptiionV3
        elif self.last_conv_layer_name == 'mixed10':
            img_array /= 2.
            img_array += 0.5
            img_array *= 255.
            img_array = np.clip(img_array, 0, 255).astype('uint8')

        # Model : ResNet50V2
        elif self.last_conv_layer_name == 'conv5_block3_out':
            img_array /= 2.
            img_array += 0.5
            img_array *= 255.
            img_array = np.clip(img_array, 0, 255).astype('uint8')

        # Model : Xception
        elif self.last_conv_layer_name == 'block14_sepconv2_act':
            img_array /= 2.
            img_array += 0.5
            img_array *= 255.
            img_array = np.clip(img_array, 0, 255).astype('uint8')

        return img_array


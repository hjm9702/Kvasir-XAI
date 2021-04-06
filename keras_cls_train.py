import numpy as np
import sys, pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

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
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as InceptionResNetV2_preprocess

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))



model_id = 1




image_dim = 384


with open('kvasir_cls_'+str(image_dim)+'.pickle', 'rb') as f:
    [X,Y] = pickle.load(f)
    print(X.shape, Y.shape)



assert np.max(X) == 255
assert np.min(X) == 0



enc = OneHotEncoder(sparse=False)
Y_enc = enc.fit_transform(Y)
n_class = Y_enc.shape[1]




X_trnval, X_tst, Y_trnval, Y_tst = train_test_split(X, Y_enc, test_size = 1200, random_state = 27407, stratify = Y_enc)
X_trn, X_val, Y_trn, Y_val = train_test_split(X_trnval, Y_trnval, test_size = 1200, random_state = 27407, stratify = Y_trnval)



print('trn.shape', X_trn.shape, Y_trn.shape)
print('val.shape', X_val.shape, Y_val.shape)
print('tst.shape', X_tst.shape, Y_tst.shape)




if model_id == 1:
    base_model = VGG19(weights = 'imagenet', pooling='avg', include_top =False)
    preprocess_func = VGG19_preprocess
elif model_id == 2:
    base_model = InceptionV3(weights = 'imagenet', pooling='avg', include_top = False)
    preprocess_func = InceptionV3_preprocess
elif model_id == 3:
    base_model = ResNet50V2(weights = 'imagenet', pooling='avg', include_top = False)
    preprocess_func = ResNet50V2_preprocess
elif model_id == 4:
    base_model = InceptionResNetV2(weights = 'imagenet', pooling='avg', include_top = False)
    preprocess_func = InceptionResNetV2_preprocess



predictions = Dense(8, activation='softmax')(base_model.output)




model = Model(inputs = base_model.input, outputs = predictions)
model.summary()




data_gen_args = dict(rotation_range = 360, width_shift_range=0.15, height_shift_range=0.15, zoom_range=0.15,
                     brightness_range=[0.5, 1.5], horizontal_flip=True, vertical_flip=False, fill_mode='constant', cval=0)

generator = ImageDataGenerator(**data_gen_args, preprocessing_function = preprocess_func)



batch_size = 10




data_flow = generator.flow(X_trn, Y_trn, batch_size = batch_size)




X_val = preprocess_func(X_val)
X_tst = preprocess_func(X_tst)



print(':: training')
for layer in base_model.layers: layer.trainable = True
model.compile(optimizer = Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])




earlystopper = EarlyStopping(monitor= 'val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor= 'val_loss', factor=0.1, patience=10, min_lr = 1e-8, verbose=1)
mcp_save = ModelCheckpoint('kvasir_cls_'+str(model_id)+'.h5', save_best_only=True, monitor='val_accuracy', mode='max')



model.fit_generator(data_flow, steps_per_epoch=np.ceil(len(X_trn)/batch_size), epochs=500, validation_data=(X_val, Y_val),
                   callbacks=[earlystopper, reduce_lr, mcp_save], verbose=2)


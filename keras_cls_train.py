import numpy as np
import sys, pickle, csv

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
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as Xception_preprocess

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # memory increase is needed before starting program
        print(e)



model_id = int(sys.argv[1])



image_dim = 384


with open('kvasir_cls_'+str(image_dim)+'no_polyp'+'.pickle', 'rb') as f:
    [X,Y] = pickle.load(f)
    print(X.shape, Y.shape)

with open('kvasir_cls_'+str(image_dim)+'polyp' + '.pickle', 'rb') as w:
    [X_polyp, Y_polyp] =  pickle.load(w)
    print(X_polyp.shape, Y_polyp.shape)

assert np.max(X) == 255
assert np.min(X) == 0
assert np.max(X_polyp) == 255
assert np.min(X_polyp) == 0


# rest dataset split
X_trnval_no, X_tst_no, Y_trnval_no, Y_tst_no = train_test_split(X, Y, test_size = 1.5 / 10, random_state = 27407, stratify = Y)
X_trn_no, X_val_no, Y_trn_no, Y_val_no = train_test_split(X_trnval_no, Y_trnval_no, test_size = 1.5 / 8.5, random_state = 27407, stratify = Y_trnval_no)

# polyp split
X_trnval_polyp, X_tst_polyp, Y_trnval_polyp, Y_tst_polyp = train_test_split(X_polyp, Y_polyp, test_size = 1.5 / 10, random_state = 27407)
X_trn_polyp, X_val_polyp, Y_trn_polyp, Y_val_polyp = train_test_split(X_trnval_polyp, Y_trnval_polyp, test_size = 1.5 / 8.5, random_state = 27407)

# concatenation and onehot encoding
enc = OneHotEncoder(sparse=False)

X_trn = np.concatenate((X_trn_no, X_trn_polyp), axis=0)
Y_trn = np.concatenate((Y_trn_no, Y_trn_polyp))
Y_trn = enc.fit_transform(Y_trn)

n_class = Y_trn.shape[1]
print("total class number: ", n_class)

X_val = np.concatenate((X_val_no, X_val_polyp), axis=0)
Y_val = np.concatenate((Y_val_no, Y_val_polyp), axis=0)
Y_val = enc.fit_transform(Y_val)

X_tst = np.concatenate((X_tst_no, X_tst_polyp), axis=0)
Y_tst = np.concatenate((Y_tst_no, Y_tst_polyp), axis=0)
Y_tst = enc.fit_transform(Y_tst)


print('trn.shape', X_trn.shape, Y_trn.shape)
print('val.shape', X_val.shape, Y_val.shape)
print('tst.shape', X_tst.shape, Y_tst.shape)



model_list = ["VGG19", "InceptionV3", "ResNet50V2", "Xception"]
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
    base_model = Xception(weights = 'imagenet', pooling='avg', include_top = False)
    preprocess_func = Xception_preprocess



predictions = Dense(8, activation='softmax')(base_model.output)


model = Model(inputs = base_model.input, outputs = predictions)
model.summary()




data_gen_args = dict(rotation_range = 360, width_shift_range=0.15, height_shift_range=0.15, zoom_range=0.15,
                     brightness_range=[0.5, 1.5], horizontal_flip=True, vertical_flip=False, fill_mode='constant', cval=0)

generator = ImageDataGenerator(**data_gen_args, preprocessing_function = preprocess_func)



batch_size = 16


data_flow = generator.flow(X_trn, Y_trn, batch_size = batch_size)

X_val = preprocess_func(X_val)
X_tst = preprocess_func(X_tst)



print(':: training {} ::::'.format(model_list[model_id - 1]))
for layer in base_model.layers: layer.trainable = True
model.compile(optimizer = Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])




earlystopper = EarlyStopping(monitor= 'val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor= 'val_loss', factor=0.1, patience=10, min_lr = 1e-8, verbose=1)
mcp_save = ModelCheckpoint('models/kvasir_cls_'+model_list[model_id - 1]+'.h5', save_best_only=True, monitor='val_accuracy', mode='max')



model.fit_generator(data_flow, steps_per_epoch=np.ceil(len(X_trn)/batch_size), epochs=500, validation_data=(X_val, Y_val),
                   callbacks=[earlystopper, reduce_lr, mcp_save], verbose=1)

# prediction
print(':: prediction')
#model = load_model('mnnet_cls_'+str(model_id)+'.h5')
Y_tst_hat = model.predict(X_tst, batch_size=batch_size)
accuracy = accuracy_score(np.argmax(Y_tst, 1), np.argmax(Y_tst_hat, 1))
print(accuracy)

#print(confusion_matrix(np.argmax(Y_tst, 1), np.argmax(Y_tst_hat, 1)))

f = open('results' + '/accuracy_' + model_list[model_id - 1] + '.csv', 'w', newline='')
with f:
    writer = csv.writer(f)
    writer.writerow([accuracy])
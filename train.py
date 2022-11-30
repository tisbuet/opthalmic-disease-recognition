import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

seed = 7
np.random.seed(seed)

def SympIndexing(i):
    switcher={
            'anterior segment image':0,
            'image offset':0,
            'low image quality':0,
            'no fundus image':0,
            'lens dust':0,
            'optic disk photographically invisible':0,
            'image offset':0,
            'normal fundus':1,
            'retinopathy':2,
            'glaucoma': 3,
            'cataract':4,
            'macular':5,
            'hypertensive retinopathy': 6,
            'myopia':7        
         }
    return switcher.get(i,8)

Annotation_file = pd.read_excel("ODIR-5K_Training_Annotations(Updated)_V2.xlsx")
image_names = np.vstack((Annotation_file['Left-Fundus'],Annotation_file['Right-Fundus'])).T
image_names = np.reshape(image_names, (np.product(image_names.shape),))

image_files = []
b = []
print('taking input...........')
count = 0
for im in tqdm(image_names):
    image_path = os.path.join("ODIR-5K_Training_Images/ODIR-5K_Training_Dataset", im)
    bgr = cv2.imread(image_path)
    bgr = cv2.resize(bgr, (32, 32))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    a = bgr[:,:,1]
    a = a/255
    b.append(a)
X_train = np.asarray(b)
X_train = X_train.reshape(-1, 32, 32, 1).astype('float32')

print('Input taken......')

givenLabels = np.asarray(np.vstack((Annotation_file['N'].values, Annotation_file['D'].values, \
                                    Annotation_file['G'].values, Annotation_file['C'].values, \
                                    Annotation_file['A'].values, Annotation_file['H'].values, \
                                    Annotation_file['M'].values, Annotation_file['O'].values))).T
givenLabels = np.asarray(givenLabels)

symptoms = []
symptoms = np.vstack((Annotation_file['Left-Diagnostic Keywords'],Annotation_file['Right-Diagnostic Keywords'])).T
symp = np.reshape(symptoms, (np.product(symptoms.shape),))

y_train = []
ID = []
for sym in symp:
    syms1 = sym.split(", ")
    syms2 = sym.split(",")
    if len(syms2) > len(syms1):
        syms = np.unique(syms2)
    else:
        syms = np.unique(syms1)
    a = 0
    D = []
    k = 0
    for s in syms:
        if s.find('normal fundus')+1:
            a = SympIndexing('normal fundus')
            if len(syms)-1:
                a = 0
                k = 1
        elif s.find('proliferative retinopathy')+1 or s.find('diabetic retinopathy')+1:
            a = SympIndexing('retinopathy')
        elif s.find('glaucoma')+1:
            a = SympIndexing('glaucoma')
        elif s.find('cataract')+1:
            a = SympIndexing('cataract')
        elif s.find('macular epiretinal membrane')+1 or s.find('macular pigmentation disorder')+1 or s.find('macular hole')+1 or s.find('macular coloboma')+1:
            a = SympIndexing(s)
        elif s.find('macular')+1:
            a = SympIndexing('macular')
        elif s.find('hypertensive retinopathy')+1:
            a = SympIndexing('hypertensive retinopathy')
        elif s.find('myopia')+1 or s.find('myopic')+1:
            a = SympIndexing('myopia')
        else:
            a = SympIndexing(s)
        D.append(a)
    D = np.unique(D)
    if 0 in D:
        D = list(filter(lambda x: x != 0, D))
    j = [0, 0, 0, 0, 0, 0, 0, 0]
    for d in D:
        j += np_utils.to_categorical(d-1,8)
    if np.sum(j)==0:
        j = [1, 0, 0, 0, 0, 0, 0, 0]
    y_train.append(j)
Y_train = np.asarray(y_train).astype('float32')

def create_model(x, y):    
    model = Sequential()
    model.add(Conv2D(8,kernel_size = (3,3), padding="same",input_shape=(x.shape[1:]), dilation_rate=3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2, data_format=None))
    
    model.add(Dropout(0.25))
    model.add(Flatten())
    
    model.add(Dense(16, activation= 'relu' ))
    model.add(Dropout(0.15))
    model.add(Dense(y.shape[1], activation='sigmoid'))

    model.compile(loss='category_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

file_path = 'best.h5'
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=40)
tb = TensorBoard(log_dir='./', histogram_freq=1, write_graph=True, write_images=False)
callbacks_list = [checkpoint, early, tb]
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.2,
    width_shift_range=0.2, height_shift_range=0.2, vertical_flip=True,horizontal_flip=True,                    
    fill_mode="nearest")

x, y = shuffle(X_train, Y_train)

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(x):
    t_x, val_x = x[train_index], x[test_index]
    t_y, val_y = y[train_index], y[test_index]

    model = None
    model = create_model(t_x, t_y)
    
    results = model.fit_generator(
        aug.flow(t_x, t_y, batch_size=32),validation_data=(val_x, val_y),
        steps_per_epoch=len(t_x)//32, epochs=50, callbacks = callbacks_list,
        shuffle = True
        )
finemodel=create_model(x, y)
finemodel.load_weights(file_path)
for layer in finemodel.layers[:-2]:
    layer.trainable = False
finemodel.fit(t_x, t_y, validation_data=(val_x, val_y), epochs=100, batch_size = 32, shuffle=True, verbose=2,\
                        callbacks = [checkpoint, early])


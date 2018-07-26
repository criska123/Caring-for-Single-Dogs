import keras
import os
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy as np
import matplotlib.pyplot as plt

def file_process():
    dir_path = "your/path/to/neg_samples"
    files = os.listdir(dir_path)
    old_name = []
    for i in files:
        if (i[-3:] == "jpg") or (i[-3:] == "png"):
            old_name.append(i)
    for i, j in enumerate(old_name):
        new_name = 'n'+str(i)+'.jpg'
        os.rename(dir_path+j, dir_path+new_name)

    dir_path = "your/path/to/pos_samples"
    files = os.listdir(dir_path)
    old_name = []
    for i in files:
        if (i[-3:] == "jpg") or (i[-3:] == "png"):
            old_name.append(i)
    for i, j in enumerate(old_name):
        new_name = 'p'+str(i)+'.jpg'
        os.rename(dir_path+j, dir_path+new_name)

def train_visualization(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

def creat_model():
    my_model = Sequential()
    my_model.add(Flatten(input_shape=(7,7,512)))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(64, activation='relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(1, activation='sigmoid'))
    return my_model

if __name__ == "__main__":
    # files process
    # file_process()

    # files name input
    datax_img = []
    nneg = 0
    npos = 0
    data_path_neg = "your/path/to/neg_samples"
    tmp = os.listdir(data_path_neg)
    for i in tmp:
        if i[-3:] == "jpg":
            nneg += 1
            datax_img.append(i)

    data_path_pos = "your/path/to/pos_samples"
    tmp = os.listdir(data_path_pos)
    for i in tmp:
        if i[-3:] == "jpg":
            npos += 1
            datax_img.append(i)
    print("number of the negative samples:", nneg)
    print("number of the positive samples:", npos)

    # creat label
    datay = [0]*nneg + [1]*npos

    # data check
    print()
    print("________data check________")
    nwrong = 0
    for i, j in zip(datax_img, datay):
        if i[0]=='n' and j!=0 or (i[0]=='p' and j!=1):
            nwrong += 1
            print(i,j)
    if nwrong:
        print("wrong data pairs above\n")
    else:
        print("data check passed\n")

    # get feature 
    i = 0
    data = None
    dir_path = "your/path/to/all_samples"
    # if you have download the net weights
    vgg16_model = VGG16(weights="your/path/to/weight", include_top=False)
    # if you haven't download the net weights
    # model = VGG16(weights='imagenet', include_top=False)
    # other models
    # xception_model = Xception(weights='D:/DD/Project/Hackthon/net_weight/xception_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
    # mobile_model = MobileNet(weights='D:/DD/Project/Hackthon/net_weight/mobilenet_1_0_224_tf_no_top.h5', include_top=False)
    model = vgg16_model
    print("_________feature generation__________")
    start = time.time()
    for img_name in datax_img:
        img_path = os.path.join(dir_path, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        start = time.time()
        # the feature shape is (None, 7, 7, 512)
        features = model.predict(x)
        if i == 0:
            datax = features
        else:
            datax = np.append(datax, features, axis=0)
        i += 1
        if i%100 == 0:
            print("have done", i)
    start = time.time()
    print("complete features counts:", i)
    print("cost time:", time.time()-start)

    vgg16_datax = datax
    datay = np.asarray(datay)

    # creat my model
    my_vgg16_model = creat_model()

    print()
    print(vgg16_datax.shape)
    print("_______training begin________")
    my_vgg16_model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    history = my_vgg16_model.fit(vgg16_datax, datay,
              epochs=80,
              batch_size=32,
              validation_split=0.1)

    # for visualization 
    train_visualization(history)

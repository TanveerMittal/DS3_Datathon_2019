import cv2
import pickle
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.applications import vgg19, inception_v3, resnet50
from sklearn.model_selection import train_test_split

num_classes = 40
epochs = 1
batch_size = 12

x = pickle.load(open("transformed.pkl", 'rb'))
print("loaded data")

df = pd.read_csv("data/train.csv")

train_names = [x[11:][:-4] for x in glob("data/train/*")]
y = np.array([df["breedID"][int(i)] for i in train_names])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
print("split data")
del x, y, df

model = load_model("vgg19.h5")

opt = optimizers.SGD(lr=0.001, decay=1e-6)

print("compiling model")
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs,shuffle=True, validation_data=(x_test, y_test))

model.save("vgg19_model.h5")

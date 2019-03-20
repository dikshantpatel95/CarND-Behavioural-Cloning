
import csv
import numpy as np
import cv2
import os

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Convolution2D
#from keras.layers.convolutional import Conv2D

def load_data(images, measurements, paths, all_data):
    lines = []
    for path in paths:
        with open(path+"driving_log.csv") as csvfile:
            lines = []
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
            #print(len(lines), path)
        # read in Data
        firstline = True    # ignore first line in case its labels
        for line in lines:
            if firstline:
                firstline = False
                continue
            source_path_center = line[0]
            filename = source_path_center.split('/')[-1]    # split file path to get file name
            if (len(filename) > 50):
                 filename = source_path_center.split('\\')[-1]    # split file name for path in windows format
            current_path = path + 'IMG/'+filename
            image = cv2.imread(current_path)   

            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
        
        # if all_data is true, use data from all three cameras
        if all_data:
            delta = 0.35    # value to add/ subtract from side cameras
            for i in range(0,2):
                firstline = True
                for line in lines:
                    if firstline:
                        firstline = False
                        continue
                    source_path = line[i+1]
                    filename = source_path.split('/')[-1]
                    if (len(filename) > 50):
                        filename = source_path.split('\\')[-1]
                    current_path = path + 'IMG/'+filename
                    image = cv2.imread(current_path)   
                    images.append(image)
                    if i == 0:    # left camera image
                        measurement = float(line[3]) + delta
                    else:
                        measurement = float(line[3]) - delta
                    measurements.append(measurement)
        print(path, "images = ", len(images))
        #print(images[4578].shape)
        firstline = True
    
    return images, measurements    # return the images and measurements as list
    
# create training sets
def create_dataset(images, measurements):
    X_train = np.array(images)
    y_train = np.array(measurements)
    return(X_train, y_train)

# NVidia neural network model
def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(64, 3, 3, activation = "relu"))
    model.add(Convolution2D(64, 3, 3, activation = "relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

# augment images to the training data set
def augment_data(imgs, msmt):
    imgs_aug = []
    msmt_aug = []
    if (len(imgs) == len(msmt)):
        for i in range(1, len(imgs)):
            imgs_aug.append(imgs[i])
            msmt_aug.append(msmt[i])
            imgs_aug.append(cv2.flip(imgs[i], 1))
            msmt_aug.append(-1 * msmt[i])
    return imgs_aug, msmt_aug

imgs_list = []
msmt_list = []

# load teh path lists to read the images from
paths = []
paths.append("sim_Data/")

# load images from all cameras- left, right, center
imgs_list, msmt_list = load_data(imgs_list, msmt_list, paths, True)   # pass 1 to load images from all cameras, 0 for only center camera

# from the images read above, flip the images to generate an augmented data set
imgs_list, msmt_list = augment_data(imgs_list, msmt_list)


# load images from only the center camera
imgs_list, msmt_list = load_data(imgs_list, msmt_list, paths, False)
print("img_list type = ", type(imgs_list))

# create training dataset
X_train, y_train = create_dataset(imgs_list, msmt_list)
#print("X_train type = ", type(X_train))
print(X_train.shape)
#X_train, y_train = c.preprocess_data(X_train, y_train)

# function to get the model
model = get_model()

# compile the model and choose adam optimizer
model.compile(loss='mse', optimizer = 'adam')

# train the model, split training, validation set
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=15)

# save the mdodel file
model.save('model.h5')



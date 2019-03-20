# Behavioral Cloning Project

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)


The Approach
---

To train the nueral network we need lots of input data. So to get input data i ran the simulator and then tested for a couple of laps on the simulator to get used to and then I recorded around 5 laps of data which amounted to around 15500 images which have images from left, right and center camera respectively.

I ran the data recording earlier but driving at steering angle 0 for longer times lead to bad outputs after the nueral network was trained hence i had to induce some inputs at regular intervals the final data set help me get the best result. 

Also i collected sufficient data in 5 laps to cover the bridge and sharp corner scenarios.

###Data Augmentation

To generate augmented data i flipped all the images. This helped me avoid recollecting all the data driving in the oppposite driection.We also had images from the left and right cameras, so I used those in addition to the front camera images. As suggested in the lectures I modified the values of the measurements associated with the right and left cameras by a value Δ=0.35 .

###Data PreProcessing :

I used normalization as a preprocessing technique on the images. In addition, I also modified the image intensities further to get the mean around 0 as suggested in the lecture videos.
Before applying the pre processing, the training and validation errors were in the range of 600-700 after 10-12 epochs. but after the pre processing, the error went down to around 0.5


###Model:
The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (and later in the student forum, the [comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py)) - the diagram below is a depiction of the nVidia model architecture.

<img src="./images/nVidia_model.png?raw=true" width="400px">

I implemented this model using the keras framework as follows:

  ```python
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
  ```
I used the Adam optimizer to optimize the output of the layers, as this is one of the most widely used optimizers in deep learning
problems. and I use the mean square error function to calculate the error.
I split the training data collected into training data and validation data. the ratio to training vs validation is 80:20

###Trials and result:
I tried out multiple configurations to best train the model like modifying the Epochs and making sure that the model does not
overfit, using more of straight driving images, more images with curves, etc.
I also tweaked the delta mentioned above to modify the measurements for images from left and right cameras to get the best result.
I got the best result with the Epochs = 15, and delta = 0.35
the training set was close to 15500 images, of which almost 80% were used for training and the rest 20% for validation.
The video is saved as video.py.

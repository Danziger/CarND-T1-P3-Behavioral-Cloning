CarND · T1 · P3 · Behavioral Cloning
====================================

[//]: # (Image References)

[image1]: ./output/images/001%20-%20All%20Signs.png "All Signs"

[sign1]: ./input/images/resized/001%20-%20Yield.jpg "Yield"


Project Goals
-------------

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior.
* Build, a convolution neural network in Keras that predicts steering angles from images.
* Train and validate the model with a training and validation set.
* Test that the model successfully drives around track one without leaving the road.
* Summarize the results with a written report.


Rubric Points
-------------

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.


### FILES SUBMITTED & CODE QUALITY

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode.

My project includes the following files:

* `model.py`: Containing the script to create and train the model. Some functionality has been extracted to separated files: `utils.py` and `constants.py`.

* `drive.py`: For driving the car in autonomous mode. The only change here has been to convert the images to YUV, as the training was done with YUV images, and increasing the default speed of 9 to either 10, 15 or 20 in order to generate the various videos with a higher driving speed.

* `models/`: Multiple videos have been provided for demonstrational purposes. The relevant one for the evaluation is `beach.h5`.

* `output/videos/`: Multiple videos have been provided for demonstrational purposes. The relevant one for the evaluation is `beach-data-beach-track-20-ok`.

* `WRITEUP.md` summarizing the results.

* `analysis.py`, `examples.py` and `plot.py` were used to generate verious images for the write up.


#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the first track by executing:

```sh
python drive.py ../models/beach.h5
```

To drive it on the second track use:
```sh
python drive.py ../models/mountain.h5
```

The other models are there just for demonstrational purposes, but can also be used with that same command.


#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, logs its layers, as well as saving a diagram of it, and it contains comments to explain how the code works.

Some of the functionality has been extracted to `utils.py` and `constants.py`, so comments can also be found there.


### MODEL ARCHITECTURE AND TRAINING STRATEGY

#### 1. An appropriate model architecture has been employed

# TODO model-ok info?

# TODO: Add NVIDIA image

As suggested through the course, my model architecture is based on the one from
[NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), with the only changes being:

- The addition of a Cropping2D layer (TODO code line 18) in order to remove irrelevant portions of the image using the GPU instead of doing
  with the processor.

- The addition of a Lambda layer (TODO code line 18) to normalize (`Lambda(lambda x: (x / 255.0) - 0.5)`) the data, again, using the GPU.

- The addition of L2 regularization and ELU activation layers (TODO starting at code line 18) to prevent overfitting
  and introduce nonlinearity, respectively. Both have been used instead of Dropout layers and RELU
  activation layers after reading some discussions on the forums/slack channels, various articles comparing different
  alternative methods and trial and error, as we will see in greater detail the following sections.


#### 2. Attempts to reduce overfitting in the model

The model contains L2 regularization in all convolutional and fully-connected layers (except the last one) in order to
reduce overfitting (TODO: model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (TODO: code line 10-16).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 25). However, the initial
learning rate have been reduced from the default 0.001 to 0.0005 (half the default).

# TODO: Explain the process 0.001 -> 0.0001 -> 0.0005

# TODO: Add code lines in general!


#### 4. Appropriate training data

New training data was recorded to keep the vehicle driving on the road, using the mouse to progressively steer it,
generating smother data. This new training data includes both normal laps and recovery data, both recorded in both
directions.

The images from all 3 cameras have been used and all that data have been augmented using various methods, from simple
ones like flipping the images horizontally and changing the sign of the angle, to more advanced ones like changing the
brightness, contrast or sharpness of the images.

It's worth mentioning that the performance of the model is still not amazing mainly due to the way the full laps were
recorded, as instead of doing center lane driving I tried to follow racing lines, which won't help the model drive
properly with this kind of neural network (it won't be able to drive properly nor on the center of the lane nor
following racing lines as I did).

For details about this issue and how I created the training data, see the next section.


### ARCHITECTURE, TRAINING DOCUMENTATION & POSSIBLE IMPROVEMENTS

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement the architecture from NVIDIA as designed by them
and add small modifications to it in order to prevent overfitting and adapt it to the project's needs.

My first step was to use a simple one-fully-connected-layer neural network model as we saw throw the course, just ot get
familiar with Keras first.

The first addition to this were the Cropping2D and Lambda layers, to remove irrelevant data from the images (hood and
sky) and normalize the data, respectively.

At this point, I was training the model for just 6 epochs and using only the center camera of the example data, as I
wasn't really trying to obtain a valid model from this architecture.

Next, I replaced my only fully-connected layer with NVIDIA's architecture, as suggested thought the course. I think this
model might be appropriate because the problem they were trying to solve is quite similar to ours and I trust NVIDIA as
a reliable source of knowledge and relevant information. I also started training for 8 epochs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and
validation set. I found that my the mean squared error would decrease quite fast and then start bouncing back and forth
continuously, so probably the initial learning rate was too big. I first tried decreasing it from the default of 0.001
to 0.0001, but it was now too slow. I finally used 0.0005 (half the default). After doing this, I also increased the
epochs from 8 to 12.

The next problem I found was that I had a low mean squared error on the training set but a high mean squared error on the
validation set. This implied that the model was overfitting. To combat it, I modified the model introducing Dropout
layers, as I did in the previous project. However, this time the difference was not as big as before and also the mean
squared error was decreasing slower, so I would have to increase the epochs. Instead, I took a look to some discussions
around that topic in the forums and Slack channels and, after reading various online articles and Medium posts about
regularization and dropout, I decided to give the former a try, so I added L2 regularization to all my convolutional and
fully-connected layers (except the last one).

This reduced the overfitting and generated a model that was driving smoothly the first long turn of the track and the
next one, but crashing into the bridge. I then decided to use the images from all 3 cameras and flip them horizontally
as well, so the dataset would now be 6 times bigger. This got the car a bit further, just across the bridge, but it
crashed in the next turn, so I decided to record my own data, including full laps and recovery data in both directions
and for both circuits.

At the end of the process, the vehicle is able to drive autonomously around both tracks, but with different models
trained specifically for each of them. When merging the data of both circuits together it will fail in both.

# TODO: Talk about the data analyris and that stuff...

# TODO: Also, after this augmentation it was worse than using example data. maybe not all augmentations are realistic or
# are not the most valuable ones (shift?)

# TODO: ELU Vanishing gradient

I tried augmenting the data in some other ways (brightness, contrast, sharpenes...) but this didn't help, so after
spending some more time on this without success, I decided to submit the old model that would only drive on the firs
track.

I will go into greater detail about this problem in the last section, as this problem is mainly related with the data
I recorded. Therefore, augmenting it won't help to generate a better model, as the original data used to create the
augmented one is already "corrupted" or just wrong.

# Shit in, shit out



#### 2. Final Model Architecture

# TODO: Polish code first, then continue with writeup adding lines and finishing following sections and then polish all, before
# adding graphics.

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]








____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           cropping2d_input_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 65, 320, 3)    0           cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________










#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

# TODO: Mention augmentation using Carla or shifting the images..

# TODO: Mention racing lines will make the model not know how to react near the border unless much more recovery data than normal data has
# been recorded, which is actually the other way around...

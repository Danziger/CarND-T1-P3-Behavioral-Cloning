CarND · T1 · P3 · Behavioral Cloning Writeup
============================================

First of all, the results of the project can be found in YouTube:

1. [Beach data on beach track at 20](https://www.youtube.com/watch?v=2aQ2ddly7Y8)
2. [Both data on beach track at 20 (crash)](https://www.youtube.com/watch?v=njMM31ynCVM)
3. [Both data on beach track at 15](https://www.youtube.com/watch?v=BdSZWsjF4zA)
4. [Both data on mountain track at 15 (crash)](https://www.youtube.com/watch?v=oX7vcNPxgU0)
5. [Both data on mountain track at 10 (crash)](https://www.youtube.com/watch?v=uaukR5QKSSk)
6. [Mountain data on mountain track at 15](https://www.youtube.com/watch?v=FRqvp0jG8tc)


[//]: # (Image References)

[image1]: ./output/images/001%20-%20NVIDIA%20CNN.png "NVIDIA CNN"
[image2]: ./output/images/002%20-%20Architecture%20Diagram.png "Architecture Diagram"

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

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how
I addressed each point in my implementation.


### FILES SUBMITTED & CODE QUALITY

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode.

My project includes the following files:

* `model.py`: Containing the script to create and train the model. Some functionality has been extracted to separated
files: `utils.py` and `constants.py`.

* `drive.py`: For driving the car in autonomous mode. The only change here has been to convert the images to YUV, as the
training was done with YUV images, and increasing the default speed of 9 to either 10, 15 or 20 in order to generate the
various videos with a higher driving speed.

* `models/`: Multiple videos have been provided for demonstrational purposes. The relevant one for the evaluation is
`both.h5`.

* `output/videos/`: Multiple videos have been provided for demonstrational purposes. The relevant one for the evaluation
is `both-data-beach-track-15-ok` (https://www.youtube.com/watch?v=BdSZWsjF4zA).

* `WRITEUP.md` summarizing the results.

* `analysis.py`, `examples.py` and `plot.py` were used to generate various images for the write up.


#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the first track by
executing:

```sh
python drive.py ../models/beach.h5
```

To drive it on the second track use:
```sh
python drive.py ../models/mountain.h5
```

The other models are there just for demonstrational purposes, but can also be used with that same command.


#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the
pipeline I used for training and validating the model, logs its layers, as well as saving a diagram of it, and it
contains comments to explain how the code works.

Some of the functionality has been extracted to `utils.py` and `constants.py`, so comments can also be found there.


### MODEL ARCHITECTURE AND TRAINING STRATEGY

#### 1. An appropriate model architecture has been employed

As suggested through the course, my model architecture is based on the one from
[NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/):

![NVIDIA CNN][image1]

The only two changes made to it have been, on one hand, the addition of a Cropping2D layer (`model.py:76`) in order to
remove irrelevant portions of the image using the GPU instead of doing it with the processor and, on the other hand, the
addition of L2 regularization and ELU activation layers (`model.py:78-88`) to prevent overfitting and introduce
nonlinearity, respectively.

ELU should make learning faster than ReLU as has a mean closer to 0 and L2 regularization is an analyticsl alternative
to dropout, which randomly turns on and off neurons while training.

It's worth mentioning that the dimensions of the layers also change in order to accomodate the generated images, which
are `160px × 320px` originally and `65px × 320px` after cropping them.

#### 2. Attempts to reduce overfitting in the model

The model contains L2 regularization in all convolutional and fully-connected layers (except the last one) in order to
reduce overfitting (`model.py:78-88`).

The model was trained and validated on different data sets to ensure that the model was not overfitting (`model.py:21`).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track, as it
can be seen in the videos listed at the top of this document.


#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (`model.py:95`). However, the initial
learning rate have been reduced from the default `0.001` to `0.0005` (half the default) by trial and error as we will
see later.

Also, the L2 regularization parameter has also been adjusted by trial and error (`model.py:78-88`).


#### 4. Appropriate training data

All the details about how the training data was obtained and augmented, as well as the issues with it and possibles ways
to improve it can be found in the last point of the next section.


### ARCHITECTURE, TRAINING DOCUMENTATION & POSSIBLE IMPROVEMENTS

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement the architecture from NVIDIA as designed by them
and add small modifications to it in order to prevent overfitting and adapt it to the project's needs.

My first step was to use a simple one-fully-connected-layer neural network model as we saw throw the course, just ot get
familiar with Keras first.

The first addition to this were the `Cropping2D` and `Lambda` layers, to remove irrelevant portions of the images (hood
and sky) and normalize the data, respectively.

At this point, I was training the model for just 6 epochs and using only the center camera of the example data, as I
wasn't really trying to obtain a valid model from this architecture.

Next, I replaced my only fully-connected layer with NVIDIA's architecture, as suggested thought the course. I think this
model might be appropriate because the problem they were trying to solve is quite similar to ours and I trust NVIDIA as
a reliable source of knowledge and relevant information. I also started training for 8 epochs.

In order to gauge how well the model was working, I split my image and steering angle data set into a training set (80%)
and a validation set (20%). I found that the mean squared error would decrease quite fast and then start bouncing back
and forth continuously, so probably the initial learning rate was too big. I first tried decreasing it from the default
of `0.001` to `0.0001`, but it was now too slow. I finally used `0.0005` (half the default). After doing this, I also
increased the epochs from 8 to 12.

The next problem I found was that I had a low mean squared error on the training set but a high mean squared error on the
validation set. This implied that the model was overfitting. To combat it, instead of using `Dropout` layers, as we did
in the last project, I decided to try L2 regularization.

This reduced overfitting and generated a model that was driving smoothly the first 2 long turn of the track, but then
crashing into the bridge. I then decided to use the images from all 3 cameras and flip them horizontally as well, so the
dataset would now be 6 times bigger. This got the car a bit further, just across the bridge, but it crashed in the next
turn, so I decided to record my own data, including full laps and recovery data in both directions and for both tracks.

At the end of the process, the vehicle is able to drive autonomously around both tracks, but with different models
trained specifically for each of them, as we can see in the videos:

- [`beach.h5`, Beach data on beach track at 20](https://www.youtube.com/watch?v=2aQ2ddly7Y8)
- [`mountain.h5`, Mountain data on mountain track at 15](https://www.youtube.com/watch?v=FRqvp0jG8tc)

TODO: Check remaining models in the folder!

Merging the data of both tracks together did actually help the model behave better in turns on the first track (although
driving speed had to be reduced), as we can see if we compare this new video with the previous one, where the car almost
missed a turn and went onto the ledge:

- [`both.h5`, Both data on beach track at 20 (crash)](https://www.youtube.com/watch?v=njMM31ynCVM)
- [`both.h5`, Both data on beach track at 15](https://www.youtube.com/watch?v=BdSZWsjF4zA)

However, it was a complete disaster in track 2:

- [`both.h5`, Both data on mountain track at 15 (crash)](https://www.youtube.com/watch?v=oX7vcNPxgU0)
- [`both.h5`, Both data on mountain track at 10 (crash)](https://www.youtube.com/watch?v=uaukR5QKSSk)

This model, `both.h5`, and the video of it driving at 15, are the ones meant to be evaluated. The rest are there just
for demonstrational purposes.

At this point I started suspecting this was a problem with the data, probably with the original data I recorded rather
than a matter of augmenting it further, as instead of doing center lane driving I was doing racing driving, following
racing lines and therefor constantly approaching curbs and even riding them, as I thought it would be possible to train
a model to drive like that.

I quickly realized in order for the model to be able to do that it would have to remember the steering angle of the
previous N states, as that's the only way to know, once you ride onto a curb, if you are about to leave the track or
just following a normal racing line. Therefore, I could either try to feed that buffer with the N last steering angles
to the network, probably appending them to the `Flatten` output, or look into Keras' recurrent neural network (RNN)
layers.

As I'm not too familiar with them and that would need quite a bit of time to change the architecture as well as the way
I feed in the training data, which would now have to be shuffled keeping the sequences together rather than as
individual images, I decided not to implement these changes and just focus on trying to get better models by further
augmenting the data, which didn't work out at the end anyway.

We will look into the data issues in detail the last section of this document.


#### 2. Final Model Architecture

The final model architecture (`model.py:72:89`) consisted of a convolution neural network with the following layers and layer sizes:

<table>
    <tr>
        <th>LAYER (NAME)</th>
        <th>IN. SIZE</th>
        <th>OUT. SIZE</th>
        <th>PARAMS</th>
        <th>DESCRIPTION</th>
    </tr>
    <tr>
        <td>Cropping2D<br/>(cropping2d_1)</td>
        <td>160 × 320 × 3</td>
        <td>65 × 320 × 3</td>
        <td>0</td>
        <td></td>
    </tr>
    <tr>
        <td>Lambda<br/>(lambda_1)</td>
        <td>65 × 320 × 3</td>
        <td>65 × 320 × 3</td>
        <td>0</td>
        <td><code>lambda x: (x / 255.0) - 0.5</code></td>
    </tr>
    <tr>
        <td>Convolution2D<br/>(convolution2d_1)</td>
        <td>65 × 320 × 3</td>
        <td>31 × 158 × 24</td>
        <td>1824</td>
        <td>24 filters, 5 × 5 kernel,<br/>2 × 2 stride & valid padding</td>
    </tr>
    <tr>
        <td>ELU<br/>(-)</td>
        <td>31 × 158 × 24</td>
        <td>31 × 158 × 24</td>
        <td>0</td>
        <td></td>
    </tr>
    <tr>
        <td>Convolution2D<br/>(convolution2d_2)</td>
        <td>31 × 158 × 24</td>
        <td>14 × 77 × 36</td>
        <td>21636</td>
        <td>36 filters, 5 × 5 kernel,<br/>2 × 2 stride & valid padding</td>
    </tr>
    <tr>
        <td>ELU<br/>(-)</td>
        <td>14 × 77 × 36</td>
        <td>14 × 77 × 36</td>
        <td>0</td>
        <td></td>
    </tr>
    <tr>
        <td>Convolution2D<br/>(convolution2d_3)</td>
        <td>14 × 77 × 36</td>
        <td>5 × 37 × 48</td>
        <td>43248</td>
        <td>48 filters, 5 × 5 kernel,<br/>2 × 2 stride & valid padding</td>
    </tr>
    <tr>
        <td>ELU<br/>(-)</td>
        <td>5 × 37 × 48</td>
        <td>5 × 37 × 48</td>
        <td>0</td>
        <td></td>
    </tr>
    <tr>
        <td>Convolution2D<br/>(convolution2d_4)</td>
        <td>5 × 37 × 48</td>
        <td>3 × 35 × 64</td>
        <td>27712</td>
        <td>64 filters, 5 × 5 kernel,<br/>2 × 2 stride & valid padding</td>
    </tr>
    <tr>
        <td>ELU<br/>(-)</td>
        <td>3 × 35 × 64</td>
        <td>3 × 35 × 64</td>
        <td>0</td>
        <td></td>
    </tr>
    <tr>
        <td>Convolution2D<br/>(convolution2d_5)</td>
        <td>3 × 35 × 64</td>
        <td>1 × 33 × 64</td>
        <td>36928</td>
        <td>64 filters, 5 × 5 kernel,<br/>2 × 2 stride & valid padding</td>
    </tr>
    <tr>
        <td>ELU<br/>(-)</td>
        <td>1 × 33 × 64</td>
        <td>1 × 33 × 64</td>
        <td>0</td>
        <td></td>
    </tr>
    <tr>
        <td>Flatten<br/>(flatten_1)</td>
        <td>1 × 33 × 64</td>
        <td>2112</td>
        <td>0</td>
        <td></td>
    </tr>
    <tr>
        <td>Dense<br/>(dense_1)</td>
        <td>2112</td>
        <td>100</td>
        <td>211300</td>
        <td></td>
    </tr>
    <tr>
        <td>ELU<br/>(-)</td>
        <td>100</td>
        <td>100</td>
        <td>0</td>
        <td></td>
    </tr>
    <tr>
        <td>Dense<br/>(dense_2)</td>
        <td>100</td>
        <td>50</td>
        <td>5050</td>
        <td></td>
    </tr>
    <tr>
        <td>ELU<br/>(-)</td>
        <td>50</td>
        <td>50</td>
        <td>0</td>
        <td></td>
    </tr>
    <tr>
        <td>Dense<br/>(dense_3)</td>
        <td>50</td>
        <td>10</td>
        <td>510</td>
        <td></td>
    </tr>
    <tr>
        <td>ELU<br/>(-)</td>
        <td>10</td>
        <td>10</td>
        <td>0</td>
        <td></td>
    </tr>
    <tr>
        <td>Dense<br/>(dense_4)</td>
        <td>10</td>
        <td>1</td>
        <td>11</td>
        <td></td>
    </tr>
</table>

                  TOTAL PARAMS: 348219
        TOTAL TRAINABLE PARAMS: 348219 (100%)
    TOTAL NON-TRAINABLE PARAMS: 0 (0%)

We can also visualize it in this diagram generated with Keras's `visualize_util`, althouth it contains a lower level of detail:

![Architecture Diagram][image2]


#### 3. Creation of the Training Set & Training Process

I first started using the example training data that was provided with the project. Once I could not generate better
models using just the center camera, I started using all 3 images (left, center and right) and their horizontally
flipped versions. This is how they look:

<table>
    <tr>
        <td colspan="3">Random sample from example data set</td>
    </tr>
    <tr>
        <td><img src="./output/images/003 - Example Image Left"</td>
        <td><img src="./output/images/004 - Example Image Center"</td>
        <td><img src="./output/images/005 - Example Image Right"</td>
    </tr>
    <tr>
        <td>Left</td>
        <td>Center</td>
        <td>Right</td>
    </tr>
</table>

<table>
    <tr>
        <td colspan="3">Random sample from example data set (flipped)</td>
    </tr>
    <tr>
        <td><img src="./output/images/006 - Example Image Left Flip"</td>
        <td><img src="./output/images/007 - Example Image Center Flip"</td>
        <td><img src="./output/images/008 - Example Image Right Flip"</td>
    </tr>
    <tr>
        <td>Left</td>
        <td>Center</td>
        <td>Right</td>
    </tr>
</table>

From the very beginning I was randomly shuffling the data and splitting it in two different sets: training (80%) and
validation (20%). The model was trained with the training set, while the validation set helped determine if the model
was over or under fitting.

As I wanted more data and data for both tracks, I started recording my own using the simulator and steering the car
using the mouse to progressively steer it, generating smother data, as we can see in this example sequence:

TODO: Add CSV

This new training data includes 4 laps on each direction on each track, but instead of doing center lane driving I was
doing racing driving, following racing lines and therefor constantly approaching curbs and even riding them, as I
thought it would be possible to train a model to drive like that. These are some examples of the images I recorded from
the middle camera:

TODO: Table with images of each track

I also recorded recovery data for both tracks, 1 lap on each direction on each of them. I started recording each
recovery from the point the car is almost going out of the track, maybe already driving onto a ledge, until it's already
more or less in the center of the road gain, so that the car would learn how to react when it's about to miss a turn.
These are some examples:

TODO: Table with images of each track

Using this data, I generated multiple models, some trained with data of just one of the tracks and some others with both
data sets merged together. Some of them were able to keep the car on the road and drive a whole lap without crashing,
but I couldn't get a single model that was able to do both tracks successfully, so I started suspecting there was a
problem with the original data I recorded rather than a matter of augmenting it further, as the data sets that I was
using with these basic augmentation techniques (flipping and using all 3 cameras) were already big enough.

I trained the models on a single track's data set for 12 epochs, and 24 epochs those on both data sets, as even thought
the loss was still decreasing, at that point it was doing that slowly and some of the models were already able to
successfully drive the car without crashing. I used an adam optimizer so that manually adjusting the learning rate
wasn't necessary, but I decreased its initial learning rate from the default `0.001` to `0.0005` (half the default) to
speed up training.

However, the performance of the model is still not amazing mainly due to the way the full laps were recorded, as instead
of doing center lane driving I tried to follow racing lines, which won't help the model drive properly with this kind of
neural network (it won't be able to drive properly nor on the center of the lane nor following racing lines as I did).

As I mentioned before, I realized I would need to use RNNs to be able to train a model to drive like that, as once you
ride onto a curb, the only way to know if you are about to leave the track or just following a normal racing line is to
consider the previous steering angles to know the trajectory you are (or have been) following.

As an example, these are two images from the data set I recorded:

TODO: Add these two examples

Both are about to leave the track, but while one was recorded as part of the full 4 laps that were meant to be the
behaviour that the model should mimic, the other one was recorded as part of the recovery laps. Therefor, the steering
angle is smoother (closer to 0) in the former, but there's no way for the model to know the difference just looking at
the images, because the relevant information in order to know if we should steer harder or softer is the trajectory that
we have been following before.

TODO: Mention racing lines will make the model not know how to react near the border unless much more recovery data than normal data has been recorded, which is actually the other way around...


For  the reasons I explained before, I decided not to implement these changes and just focus on trying to get better
models by analyzing the data and further augmenting it, which didn't work out at the end anyway. We will see why next.


##### Understanding the Data

First, I got some basic stats and histograms of the data:

TODO: Add stats and plots


##### Further Augmenting the Data

In order to get a flatter histogram, I had to either not use all the over-represented samples in each epoch or augment
the under-represented ones in some other ways, which is what I did, using any of these methods or a combination of them:

- Increase/decrease brightness in all the image, left half, right half or both half in different ways.
- Increase/decrease contrast in all the image, left half, right half or both half in different ways.
- Increase the sharpness of the image.
- Blur the image.

These are some examples of augmented images using these methods:

TODO: Add images

However, this didn't work out well, and new models trained using these augmentation techniques, on either a single
track's data or on both's, for 16-32 epochs, were not able to keep the car on the road. As we have already seen, the
original data was recorded with a behaviour in mind that can't be learned by this type of neural network, so it's not
the best data to generate a model that center lane drives the car properly.

TODO: training/validation changes as validation is not augmented

Therefore, augmenting it won't help to generate a better model, as the original data used to create the it is already
"wrong". In other words, shit in, shit out.


##### Other Augmentation

Anyway, the augmentation methods used could also be improved. For example, blurring the images might not help the model
as any of the images captured by car when driving autonomously look like that. Instead, blurring could have been used on
all the images as a postprocessing step in order to remove noise.

Moreover, in order to give the model a wider variety of steering angles to help it generalise better, the images could
have been augmented by shifting them horizontally and adjusting the steering angle based on that.


##### Alternatives to Obtain Training Data


TODO: Mention augmentation using Carla or shifting the images..

# **Behavioral Cloning**

## Writeup Report

### By Ibrahim Almohandes, April 16, 2017
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture.png "Model Visualization"
[image2]: ./examples/center_2016_12_01_13_32_42_143.jpg "Center Camera Image"
[image3]: ./examples/left_2016_12_01_13_32_42_143.jpg "Left Camera Image"
[image4]: ./examples/right_2016_12_01_13_32_42_143.jpg "Right camera Image"
[image5]: ./examples/figure_1.png "Train/Valid MSE vs no. of epochs"
[video1]: ./video.mp4 "Autonomous Driving Video"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 which shows the car being driven in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 100 (model.py lines 80-101) 

The model includes RELU layers to introduce nonlinearity (code lines 88-90 and 93-94), and the data is normalized in the model using a Keras lambda layer (code line 86). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 91, 95, and 100). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 26-65, 68-69, and 105-109). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Initially, I attempted collecting the training data myself (by running the simulator in training mode) and applying a combination of the following techniques: center lane driving, recovering from the left and right sides of the road, and driving in the opposite direction.

Then after several of these attempts without a satisfying outcome, I decided to use the training data provided by Udacity. It proved good enough for me, and it actually saved me a tremendous amount of time, as transferring the training data (as ZIP archive) from my local machine to my AWS-GPU instance (where I run the training script model.py) was very slow. Hence, defeating the purpose of using an AWS-GPU instance for speeding up the training process in the first place!

For details about how I created the training data, see the next section. 

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture is described as below.

My first step was to start with a well tested convolutional neural network model, hence I chose the NVIDIA's CNN model. I thought this model might be appropriate because NVIDIA is using it for the actual training of its autonomous vehicle(s). 

In order to gauge how well the model was working, I split my image and steering angle data into training and validation sets (80%/20% split). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it contains Dropout layers at different points between the NVIDIA's original layers.

In addition, I augmented the training data by adding left and right camera images with steering corrections of +0.2 and -0.2 respectively. This not only provided three times the training data, but also helped the car stay closer to the center of the lane as much as possible.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I tried changing both batch size and number of epochs. And even for the same batch size and number of epochs, I was getting different results!

One technique that helped the model run faster and with less memory is using a generator function and training the model with it, using the fit_generator() function.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

I would like to mention that I tried data collection techniques like driving on the opposite site of the track, and stepping from the left and right sides of the road back into the track (at points where the car went off-road), but, as I described earlier, it was a very slow process due to the low speed of data transfer, hence I used the training data provided by Udacity and focussed on improving the model instead. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 80-101) consisted of a convolution neural network with the following layers and layer sizes:

1. Three 2D Convolution layers, each with 5x5 filter size, 2x2 stride, and output sizes of 24, 36, and 48 respectively. RELU activation is applied to all three layers.
2. A dropout layer with 25% drop rate.
3. Two 2D Convolution layers, each with 3x3 filter size, default (1x1) stride, and output size of 64. RELU activation is applied to both layers.
4. Another dropout layer with 25% drop rate.
5. A flatten layer.
6. 3 dense layers with output sizes of 100, 50, and 10 respectively.
7. A third dropout layer with 50% drop rate.
8. A final dense layer with output size of 1.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used the training data provided by Udacity.

To augment the data set, I first tried flipping images and angles thinking that this would improve the training of the model, but that didn't work well, as the track is already biased towards counter-clockwise driving, so I abandoned this idea.

Then, I tried adding left and right camera images with a steering correction of +0.2 and -0.2 respectively. That proved more effective than adding the flipped image and increased the training data to triple the original. Hence I chose this augmentation technique. For example, here are three images from center, left, and right cameras (taken from the same data sample):

![alt text][image2]
![alt text][image3]
![alt text][image4]

After the collection process (using the Udacity data set), I had 8036 data points. I initially randomly shuffled the data set and put 20% of the data into a validation set. 

Then I trimmed the images to the drivable portion of the road, by cropping 60 pixels from the top (for trees, sky, etc.) and 20 pixels from the bottom (for steering while). This helped the model focus on the drivable portion of the road, hence improved the training process.

Finally, I preprocessed this data by normalizing it to the range [-1, 1] with zero mean by applying the formula ```x = x/127.5 - 1```.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the following chart.

![alt text][image5]

I used an Adam optimizer so that manually training the learning rate wasn't necessary.

In addition, I used a generator function that returns data to the Keras' fit_generator() function in batches rather than passing the whole data set for every epoch. This led to a better speed and much smaller memory footprint.

I used an AWS instance with a GPU to train the model, then downloaded the saved model (model.h5) to my local machine.

Finally, I created the autonomous driving video by running the simulator in autonomous mode in parallel with ```python drive.py model.h5 run1```, then running ```python video.py run1 --fps=40```. This created the output video (which I later renamed to video.mp4). Notice I used 40 fps for a better observation of the car behavior.

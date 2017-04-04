#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model_tn0.png "Original Nvidia Model Training Log"
[image2]: ./images/model_tn1.png "Modified Nvidia Model Training Log"
[image3]: ./images/model_nvidia1.png "Modified Nvidia Model Code"
[image4]: ./images/center.jpg "Illustration of Center Driving"
[image5]: ./images/model_v0.png "Data Testing Model Code"
[image6]: ./images/recover_0.jpg "Side recovery 0"
[image7]: ./images/recover_1.jpg "Side recovery 1"
[image8]: ./images/recover_2.jpg "Side recovery 2"
[image9]: ./images/flip_before.jpg "Image before flipping"
[image10]: ./images/flip_after.jpg "Image after flipping"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 record of testing autonomous drive by the trained model
* videos/ containing videos of tried models
* images/ containing images used in this report

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model developed by Nvidia in this [paper](https://arxiv.org/pdf/1604.07316v1.pdf) is used for this project. The neural network consists of five convolutional layers. The first three layers use a 5x5 kernel with a 2x2 stride and 24, 36, 48 filters arranged in an increasing fashion. The following two convoluational layers both use a 3x3 kernel with 64 filters each. Each of the above convoluational layer is followed by a ReLU layer as activation function for non-linearity.

Four fully-connected are included after the above block of convolutional layers, with 100, 50, 10 and 1 hidden neurons arraged in a decreasing fashion.

####2. Attempts to reduce overfitting in the model

The model contains four dropout layers in order to reduce overfitting (model.py lines 166, 168, 170, 172). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 186-190). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 33, 201).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The data is a combination of five recording runs, each in a different way which would be explained in the following section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to adjust a well-developed one by Nvidia in this [paper](https://arxiv.org/pdf/1604.07316v1.pdf). I thought this model might be appropriate because it is developed specifically for self-driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in a 4:1 ratio. The first model, which uses the original architecture developed by Nvidia, was trained with 10 epochs of data. The model had a low mean squared error on the training set but a high mean squared error on the validation set. This indicates overfitting.

![alt text][image1]

The result of simulator testing of the first model is recorded (see 
"vidoes/run_n0.mp4"). The car drives inside the lane throughout the two laps. However, the drive was mostly unstable in the sense that the car shook a lot. At 0:43 one can also see that the steering was late.

To combat overfitting, the model is modified so that a dropout layer is added before each of the fully-connected layers. The model is then trained using the same setting.

![alt text][image2]

The testing result of the modified model can be viewed in "video.mp4". One can easily see that the shaking problem is greatly mitigated. The steering at curves is also much smoother than the first model, the improvement is especially significant at 0:42 time point.

####2. Final Model Architecture

The final model architecture (model.py lines 153-175) consisted of a convolution neural network with the following layers and layer sizes:

|         Layer         |           Details          |
|:---------------------:|:--------------------------:|
| Keras Cropping Layer  | input_shape: (160, 320, 3), cropping: top = 60 pixels, botton = 25 pixels, output: (75, 320, 3)|
| Keras Lambda Layer| Scale and translate pixel values to the interval [-1, 1]|
| Convolutional Layer | filters: 24, kernel: 5x5, stride: 2x2, padding: valid, input: (75, 320, 3), output: (36, 158, 24)|
| Activation | ReLU |
| Convolutional Layer | filters: 36, kernel: 5x5, stride: 2x2, padding: valid, input: (36, 158, 24), output: (16, 77, 36)|
| Activation | ReLU |
| Convolutional Layer | filters: 48, kernel: 5x5, stride: 2x2, padding: valid, input: (16, 77, 36), output: (6, 37, 48)|
| Activation | ReLU |
| Convolutional Layer | filters: 64, kernel: 3x3, stride: 1x1, padding: valid, input: (6, 37, 48), output: (4, 35, 64)|
| Activation | ReLU |
| Convolutional Layer | filters: 64, kernel: 3x3, stride: 1x1, padding: valid, input: (4, 35, 64), output: (2, 33, 64)|
| Activation | ReLU |
|Flatten Layer| input: (2, 33, 64), output: 4224 |
|Dropout Layer| keep_prob: 0.5 |
|Fully-Connected Layer | hidden: 100 |
|Dropout Layer| keep_prob: 0.5 |
|Fully-Connected Layer | hidden: 50 |
|Dropout Layer| keep_prob: 0.5 |
|Fully-Connected Layer | hidden: 10 |
|Dropout Layer| keep_prob: 0.5 |
|Fully-Connected Layer | hidden: 1 |

The code is as follows:

![alt text][image3]


####3. Creation of the Training Set

To capture good driving behavior, one clockwise and one counter-clockwise lap for bath tracks were recorded using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

To augment the data set, I also flipped images and angles to reduce orientation biases of the data. The following is a example of images being flipped:

![alt text][image9] ![alt text][image10]

A test network is trained to test the completeness of data. The following code shows the architecture:

![alt text][image5]

The result of the simulator testing can be viewed at "videos/run1.mp4". As one can see at 0:17-0:22 the car is very close to the border, but it did take a sharp steer to recover to the center. Another problem is that near the end of the bridge the car failed to recover to the center and went out of the lane a little bit. At 0:35-0:37 the car didn't steer until very late. These three points indicate that additional data for recovering to the center of the lane is required. Therefore, I further recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from sides. These images show what a recovery looks like:

![alt text][image6] ![alt text][image7] ![alt text][image8]

The testing model is trained on this modified data set and the simulator testing result is provided at "videos/run2.mp4". One can see that the network's ability to stay in the center of the lane has improved quite a lot. 

At this stage, we can tell that the data is indeed complete as evidenced by the performance of this simple network.

After the collection process, I had 29217 images. To optimize storage efficiency, the driving log will first be processed such that each record would be converted into a tuple (name, angle, throttle, brake, speed, flipped), where "flipped" is a flag to tell the data generator whether it should flip both the image and the angle or not. Therefore, the actual number of data points that would be used in training is 29217 x 2 = 58434 data points.

####4. Training Details

The data is randomly shuffled and 20% of the data is assigned into a validation set. The remaining 80% of the data is used for training the model. The validation set helped determine if the model was over or under fitting.

The ideal correction value for left and right images is found to be around 0.2. I noticed that if the correction value is too high, the model would become too sensitive to lane borders and fails to drive straight due to constant "recovery" from sides. On the other hand, if the correction value is too low, the recovery is not sensitive enough that it may go out of the lane.

The ideal number of epochs is around 10 by experiments. After 10 epochs, very little improvement of validation loss was observed.

The adam optimizer is employed so the manual tuning of the learning rate wasn't necessary.

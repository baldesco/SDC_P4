# Self-Driving Car Engineer Nanodegree

## Project 4: Behavioral Cloning

Author: Eduardo Escobar

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

The code for this project can be found at the **model.py** file.


[//]: # (Image References)

[image1]: ./images/model.jpg "Model's architecture"
[image2]: ./images/car_view.jpg "Car's POV during simulation"
[image3]: ./images/track_2.jpg "Car's POV during simulation, track two"

___
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `run1.mp4` and `run2.mp4` containing videos of the car driving autonomously at tracks one and two, respectively

#### 2. Submission includes functional code
Using the Udacity provided simulator and the `drive.py` file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Using the simulator to record data 

To capture good driving behavior, I first recorded two laps on track one of the simulator using center lane driving. Here is an example image of center lane driving:

![alt text][image2]


Then I recorded an additional lap on track one but driving in the opposite direction. This was done to get data from a "different" road. In a similar manner, I recorded some parts of track two to have more and more diverse data.

![alt text][image3]

After recording these laps, I had a dataset of 5752 moments in time, each with a left, right and center image, and a corresponding steering angle.

#### 2. Data augmentation and use of side cameras

To create the training set, I wrote a function that collects the information in the csv file with the paths of the image files and their corresponding steering angles and appends the images and the angles to two separate lists (lines 31-44 of `model.py`). This function has the option to augment the data by flipping each image horizontally and multiplying the steering angle by -1. Doing this allows to double the number of data, and reduce the bias caused by the tracks (having more turns in one direction than the other).

Additionally, images from the side (left and right) cameras were also included in the training set, allowing to have the triple of images, compared to only using the center images. In order to include these images, a correction_factor of 0.2 was applied to their steering angles (lines 58-63 of `model.py`).

By augmenting the data and using side cameras, the dataset grew to have a total of 34512 images.

#### 3. Model Architecture

The model used for this proyect is based on the [NVIDIA End-to-End model](https://arxiv.org/abs/1604.07316). This architecture, shown in the image bellow, consists of an input and normalization layer, 5 convolutional layers, 3 fully connected hiden layer and 1 fully connected output layer.

![alt text][image1]

The architecture of this model is defined in the lines 73-90 of `model.py`. However, some modifications were made to the architecture shown in the image. 

In the first place, the dimensions of the images entering the neural network are different. In my case, the images have initial dimentions of *(160,320,3)*. These are the original dimensions of the images produced by Udacity's car simulator. When entering the model, these images are first normalized (`model.py`, line 75), and then they are cropped, removing parts of the image that are not useful to the model, such as the sky and the car hood (line 76). After this cropping operation, the dimensions of the images are *(80,320,3)*.

Additionally, activation functions are added to the convolutional and fully connected layers, in order to introduce nonlinearities in the model and capture more complex patterns. A combination of Rectified Linear Units (ReLU) and Exponential Linear Units (ELU) were chosen as the activation functions in the model, since these functions usually give good model performance and are fast to compute. It is important to note that the output layer does not have any activation function, since this is a regression model and the objective is to have an output as close as possible to the steering angles observed during the recording process.

#### 4. Attempts to reduce overfitting in the model

This architecture contains a large number of parameters and several nonlinearities, so it is important to watch out for overfitting. Two measures were taken to reduce overfitting. First, three dropout layers were included in the model's architecture, between the fully connected layers (`model.py` lines 85, 87 and 89). 

Also, 20% of the training data was used for validation of the model's performance. At each epoch the train and validation values for the loss function were monitored. Finally, at each training epoch the data was shuffled, so the order of the images did not have any influence on the model.

#### 5. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 93). As for the loss function, the mean squared error (MSE) was chosen as the function to minimize.

Through experimentation, I found that five epochs gave the best model performance, while avoiding large overfitting.

#### 6. Results

The final values obtained for the loss function were 0.0526 and 0.1301 at the train and validation data, respectively. The performance of the car at track one can be seen in the video `run1.mp4`. There, the car successfully stays on the road for a full lap. Some improvements could still be made to the way the car drives, such as smoothing the steering a little bit, so the car has a steadier course. Also, the car drives at 9 MPH in average. It would be nice to increase its speed; however, we are not controlling the car's speed yet in this project, and another model and/or methodology would be necessary to do so.

<img src="images/1.gif" width="600">

On the other hand, the car is not able to complete a full lap on track two. It fails to take a very sharp turn (it can be seen in the video `run2.mp4`). More images from this track and further training are necessary to overpass this issue.

<img src="images/2.gif" width="600">

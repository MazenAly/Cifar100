##Datasets  
The CIFAR-10 and CIFAR-100 are labeled subsets of 80 million tiny images. Both datasets consist of 60000 32x32 colour images. CIFAR-10 comprises 10 classes, with 6000 images per class whereas CIFAR-100 comprises 100 classes containing 600 images each [1].

The 100 classes in the CIFAR-100 are grouped into 20 super-classes. Each image is associated with a "fine" label (the class it is associated with) and a "coarse" label (the superclass it is associated with) [1]. These super-classes and their corresponding sub-classes are tabulated below:

##Problem description
Generally, automatically identifying objects in photographs is difficult because of the near infinite number of permutations of objects, positions, lighting, etc. possible.

Obtaining a high classification accuracy on CIFAR-100 is a challenging task because there are many classes but the number of training samples for each class is very small.  In particular, the dataset comprises only 500 training images and 100 testing images per class. It is also interesting to note that the images are of low quality and in some images, only parts of the object are shown e.g. only head or only body.

Since the same set of images is used for CIFAR-10 as well as for CIFAR-100, it is easy to see that obtaining a high classification accuracy on the latter is a more challenging task.


##Implementation
We have made use of TFLearn [2], a library built on top of TensorFlow to provide a higher-level API for facilitating and accelerating experiments for deep neural networks. TFLearn is fully compatible with TensorFlow as all its functions are built using tensors.

By using TFLearn, time-consuming and elaborate definitions of models are made compact and easy-to-understand. This is achieved by making use of TFLearn ‘layers’ (e.g. convolution and max-pool layers) which represent an abstract set of operations to build networks. Consequently, many parameters (such as bias and initial weights) are already self managed, making it easier to train a model.

We also made use of Tensorboard [3], a web application for inspecting and understanding TensorFlow runs and graphs. 

##Data loading
The dataset is downloaded from University of Toronto’s website [4]. The training data of CIFAR-100 is provided in a single file but the training data for CIFAR-10 is provided in 5 files as well as a file for the testing data. 
We unpickled and transformed the training data into a 4-dimensions tensor (number of records, image width, image length, number of channels). In addition, we put the labels of the testing data in the one-hot encoding format. 


##Image pre-processing
When training a model, the following two TFLearn defined pre-processing methods are applied at both training and testing time.

add_featurewise_zero_center: Zero center every sample with the mean (calculated over all samples).

add_featurewise_stdnorm: Scale each sample by the standard deviation (calculated over all samples).

The main reason for scaling the feature values is that we do not want one or more feature(s) to distort the learning process if a value is large compared to those of other features. In particular, it is desired for each feature of the image to have a similar defined range so that the gradients that are back-propagated to train the model do not yield erroneous results.




##Image augmentation
When training a model, the following two TFLearn defined augmentation methods are applied at training time only.

add_random_flip_leftright: Randomly flip an image (left to right).

add_random_rotation (max_angle=15.0): Randomly rotate an image by a random angle (-max_angle, max_angle).

Image augmentation is the process of generating more training samples by transforming training images, with the target of improving the accuracy and robustness of classifiers. The motivation for doing this is that in reality, objects are likely to vary in their positions in images.

This process is very important, especially in the case of CIFAR-100 since the number of training samples for each class is very small, consequently, inflating the data by augmentation is a technique used to handle to this problem [5, 6].

We augmented the training images in this project by randomly flipping each image to the right or left. In addition, we randomly rotated the images with a maximum angle of 30 degrees (maximum 15 degrees on either side). 


##Convolutional network building


Our convolutional network consists of 3 convolutional layers, each layer uses a filter of size 3x3 and it uses a ReLU activation function to capture the nonlinearity. In addition, we use a stride of 1 step and pad the input matrix with zeros around the border, so that we can apply the filter to the bordering elements of the input image matrix.

We also make use of max pooling layers of size 2x2 to help us arrive at a scale invariant representation of the image. Hence, we can detect objects in an image regardless of where they are located.

We also use 2 fully connected layers, the first one of size 600 nodes with ReLU activation function, and the second and last one is to get the probabilities of each class and we use softmax function. 

To mitigate overfitting, we use a drop out of 50%. 

We use the cross entropy cost function and we use Backpropagation to calculate the gradients of the error with respect to all weights in the network. The Adam optimizer is used to update all filter values, weights and parameter values to minimize the output error.

Tensorboard helped us visualize the architecture of the Neural Network.


##Results 
We trained the network using CIFAR-100 and CIFAR-10 data sets in three configurations, the first configuration is without data preprocessing nor data augmentation. The second, is without augmentation only. The third is using data preprocessing and augmentation. 

Cifar-100 reached 50% accuracy on the testing data while Cifar-10 reached 82%


##References
[1]  Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009.

[2] Tflearn, TensorFlow Deep Learning Library, accessed on January 9th, 2017: http://tflearn.org/

[3] TensorBoard: Visualizing Learning for TensorFlow, accessed on January 9th, 2017: https://www.tensorflow.org/how_tos/summaries_and_tensorboard/

[4] CIFAR, University of Toronto, accessed on January 9th, 2017: https://www.cs.toronto.edu/~kriz/cifar.html

[5] Benjamin Graham, Fractional Max-Pooling, Dept of Statistics, University of Warwick, CV4 7AL, UK, May 13, 2015.

[6] Benjamin Graham, Spatially-sparse convolutional neural networks, Dept of Statistics, University of Warwick, CV4 7AL, UK, September 23, 2014.


Authors: Arzam Muzaffar, Mazen Aly 




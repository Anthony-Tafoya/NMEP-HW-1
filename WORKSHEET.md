# HW 1 Worksheet

---

This is the worksheet for Homework 1. Your deliverables for this homework are:

- [ ] This worksheet with all answers filled in. If you include plots/images, be sure to include all the required files. Alternatively, you can export it as a PDF and it will be self-sufficient.
- [ ] Kaggle submission and writeup (details below)
- [ ] Github repo with all of your code! You need to either fork it or just copy the code over to your repo. A simple way of doing this is provided below. Include the link to your repo below. If you would like to make the repo private, please dm us and we'll send you the GitHub usernames to add as collaborators.

`YOUR GITHUB REPO HERE (or notice that you DMed us to share a private repo)`

## To move to your own repo:

First follow `README.md` to clone the code. Additionally, create an empty repo on GitHub that you want to use for your code. Then run the following commands:

```bash
$ git remote rename origin staff # staff is now the provided repo
$ git remote add origin <your repos remote url>
$ git push -u origin main
```
# Part -1: PyTorch review

Feel free to ask your NMEP friends if you don't know!

## -1.0 What is the difference between `torch.nn.Module` and `torch.nn.functional`?

`nn.module is the base class for building PyTorch neural networks. I helps manage parameters and provides the class structure to define our own layers. The nn.fucntional is a package that has many packages like activation and pooling functions that help make up the forward(). `

## -1.1 What is the difference between a Dataset and a DataLoader?

`The difference bewteen a dataset and a dataloader is that a Dataset groups data from various sources together, while DataLoader takes a Dataset and turns it into an iterable with a set batch size to be processed.`

## -1.2 What does `@torch.no_grad()` above a function header do?

`Torch.no.grad() means that PyTorch's gradient will not be run on the set of code and therefore cannot be used for backpropagation. Can help with speed. `

# Part 0: Understanding the codebase

Read through `README.md` and follow the steps to understand how the repo is structured.

## 0.0 What are the `build.py` files? Why do we have them?**

`The build.py files are essentiallly the middle main between the configs and working model. The goal is for the other files to use the model and data without needing to worry about intricate configs or deoendencies.`

## 0.1 Where would you define a new model?

`You would define the model in the build.py file and if need be change the model architecture in the config/model (lenet.py) file.`

## 0.2 How would you add support for a new dataset? What files would you need to change?

`You would have to change data/build.py, since that is where the configuration is for the model`

## 0.3 Where is the actual training code?

`You see the actual training implementation in the main.py file.`

## 0.4 Create a diagram explaining the structure of `main.py` and the entire code repo.

Be sure to include the 4 main functions in it (`main`, `train_one_epoch`, `validate`, `evaluate`) and how they interact with each other. Also explain where the other files are used. No need to dive too deep into any part of the code for now, the following parts will do deeper dives into each part of the code. For now, read the code just enough to understand how the pieces come together, not necessarily the specifics. You can use any tool to create the diagram (e.g. just explain it in nice markdown, draw it on paper and take a picture, use draw.io, excalidraw, etc.)

<img width="401" alt="Screen Shot 2023-03-05 at 2 10 08 PM" src="https://user-images.githubusercontent.com/87080582/222988723-b56c912b-c068-41de-8f9a-fdf1f718bc6c.png">

# Part 1: Datasets

The following questions relate to `data/build.py` and `data/datasets.py`.

## 1.0 Builder/General

### 1.0.0 What does `build_loader` do?

`Build loader takes in an argument relating to the dataset to pull. Afterward, the training, validation, and evaluation subsets are created with proper configs. For Image Net there is an augnment function which helps create artificial data to help increase the size.`

### 1.0.1 What functions do you need to implement for a PyTorch Dataset? (hint there are 3)

`For a PyTorch dataset, one needs to implement the _getitem_, _len_, and _get_transforms functions. The transforms are used to cut, crop, and rotate images so that there is more input in the data.`

## 1.1 CIFAR10Dataset

### 1.1.0 Go through the constructor. What field actually contains the data? Do we need to download it ahead of time?

`The CIFAR10Dataset (Dataset) constructor captures the data using the CIFAR10 method inside of the dataset instance var. Because you are taking it from  a directory it does seem to require some pre-loading before hand, or at the least a function that contains the location of the data. `

### 1.1.1 What is `self.train`? What is `self.transform`?

`Self.transform povides the transformation the image will undergo, which will depend on whether it is training. The Self.train indicates that the model is currently in training, meaning it will now update gradients.`

### 1.1.2 What does `__getitem__` do? What is `index`?

`_get_item takes the dataset and returns the index element in the set after it has been transformed.`

### 1.1.3 What does `__len__` do?

`_len_ returns the length of the data that we are working with.`

### 1.1.4 What does `self._get_transforms` do? Why is there an if statement?

`The self.get_transform determines the transformation the image will undergo which it is called in _getitem_. If it is in training, we will use data augmentation which include the colorjitter and flipping to provide different perspectives of the same image increasing our accuracy. If it is not training, then we have no data augmentation and just normalization (which makes data easier to navigate) and resizing to standardize the input.  `

### 1.1.5 What does `transforms.Normalize` do? What do the parameters mean? (hint: take a look here: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

`Mathematically to normalize an image is to subtract the input by the mean and put it over the std. Holistcally, it is just making the data easier to navigate and reduce the effect of outliers.`

## 1.2 MediumImagenetHDF5Dataset

### 1.2.0 Go through the constructor. What field actually contains the data? Where is the data actually stored on honeydew? What other files are stored in that folder on honeydew? How large are they?

`The file instance variable contains the data. The filepath of the data in honeydew is /data/medium-imagenet/medium-imagenet-nmep-96.hdf5. The CIFAR10 dataset is also in the same folder with the size of ___.`

> *Some background*: HDF5 is a file format that stores data in a hierarchical structure. It is similar to a python dictionary. The files are binary and are generally really efficient to use. Additionally, `h5py.File()` does not actually read the entire file contents into memory. Instead, it only reads the data when you access it (as in `__getitem__`). You can learn more about [hdf5 here](https://portal.hdfgroup.org/display/HDF5/HDF5) and [h5py here](https://www.h5py.org/).

### 1.2.1 How is `_get_transforms` different from the one in CIFAR10Dataset?

`Instead of just checking if it is trained and doing two different outcomes, getTranforms checks if it is in training mode and wants to be augmented. In addition, the transforms are kept in a list. In addition we divide by 256, which is a normalization tactic. `

### 1.2.2 How is `__getitem__` different from the one in CIFAR10Dataset? How many data splits do we have now? Is it different from CIFAR10? Do we have labels/annotations for the test set?

`This _getitem_ is different because it checks if the model is testing. If it is, then we add labels, but if not, the labels become arbitrary. In addition, because these images are made of pixel values, they are split, while the CIFAR10 was not.` 

### 1.2.3 Visualizing the dataset

Visualize ~10 or so examples from the dataset. There's many ways to do it - you can make a separate little script that loads the datasets and displays some images, or you can update the existing code to display the images where it's already easy to load them. In either case, you can use use `matplotlib` or `PIL` or `opencv` to display/save the images. Alternatively you can also use `torchvision.utils.make_grid` to display multiple images at once and use `torchvision.utils.save_image` to save the images to disk.

Be sure to also get the class names. You might notice that we don't have them loaded anywhere in the repo - feel free to fix it or just hack it together for now, the class names are in a file in the same folder as the hdf5 dataset.

`Written but not completely able to run`

# Part 2: Models

The following questions relate to `models/build.py` and `models/models.py`.

## What models are implemented for you?

`The LeNet and ResNet18 models have been implemented for me. `

## What do PyTorch models inherit from? What functions do we need to implement for a PyTorch Model? (hint there are 2)

`We inherit from the nn.module class which ensure that graidents and parameters work accordingly. We have the __init__ class and the forward() class that we inherit from.`

## How many layers does our implementation of LeNet have? How many parameters does it have? (hint: to count the number of parameters, you might want to run the code)

`We have 7 layers (2 Conv2d, 2 AvgPool2D, and 3 Linear layers). After running the code, there are 276 M parameters.`

# Part 3: Training

The following questions relate to `main.py`, and the configs in `configs/`.

## 3.0 What configs have we provided for you? What models and datasets do they train on?

`There are quite a large amount of configs, but just to name a few: color_jitter, batch_size, dataset, img_size, num_workers, pin_memory, epochs, LR, and more. Based on on our congif, we can choose to either train the model on lenet or resnet18.`

## 3.1 Open `main.py` and go through `main()`. In bullet points, explain what the function does.

`- Prepares and stores key features of model (model architecture, training/eval/val training dataset, optimizer, and loss function)`
`- Iterates over training data, running the training set, which updates gradients, and then running a validation set which can help with overfitting`
`- It keeps track of the max accuracy and then afterward runs the evaluation dataset and then saves those predictions into a csv/kaggle format`

## 3.2 Go through `validate()` and `evaluate()`. What do they do? How are they different? 
> Could we have done better by reusing code? Yes. Yes we could have but we didn't... sorry...

`Validate():`
`- Validate prepares the criterion`
`- Iterates through validation set recording loss and accuracy over the set`
`- Returns the average of the accuracy and the loss`

`Evaluate():` 
`- Sets the model into evaluation mode`
`- For each element in the training set, append the outpit to a list called preds or predictions. This is then returned.`
`- The purpose is to test the model on a set it has not seen before`

# Part 4: AlexNet

## Implement AlexNet. Feel free to use the provided LeNet as a template. For convenience, here are the parameters for AlexNet:

```
Input NxNx3 # For CIFAR 10, you can set img_size to 70
Conv 11x11, 64 filters, stride 4, padding 2
MaxPool 3x3, stride 2
Conv 5x5, 192 filters, padding 2
MaxPool 3x3, stride 2
Conv 3x3, 384 filters, padding 1
Conv 3x3, 256 filters, padding 1
Conv 3x3, 256 filters, padding 1
MaxPool 3x3, stride 2
nn.AdaptiveAvgPool2d((6, 6)) # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
flatten into a vector of length x # what is x?
Dropout 0.5
Linear with 4096 output units
Dropout 0.5
Linear with 4096 output units
Linear with num_classes output units
```

> ReLU activation after every Conv and Linear layer. DO **NOT** Forget to add activatioons after every layer. Do not apply activation after the last layer.

## 4.1 How many parameters does AlexNet have? How does it compare to LeNet? With the same batch size, how much memory do LeNet and AlexNet take up while training? 
> (hint: use `gpuststat`)

`We could use gpuststat, but if we were to do it by hand here it is how I think we would do it: 
 There are three input parameters so every input layer is going to be multiplied by 3.
 
 For the first convolution, it is (11 x 11 x 3 x 64) + 64 
 For the second convoliution, it is (5 x 5 x 3 x 192) + 192
 For the third convolution, it is (3 x 3 x 3 x 384) + 384
 For the fourth convolution, it is (3 x 3 x 3 x 256) + 256 
 For the fifth convolution,  it is (3 x 3 x 3 x 256) + 256 
 
 The linear layers are going to be: 
 (4096 x 4096) + 4096 

 The MaxPool layers have no parameters so there is no worries about that
`

## 4.2 Train AlexNet on CIFAR10. What accuracy do you get?

Report training and validation accuracy on AlexNet and LeNet. Report hyperparameters for both models (learning rate, batch size, optimizer, etc.). We get ~77% validation with AlexNet.

> You can just copy the config file, don't need to write it all out again.
> Also no need to tune the models much, you'll do it in the next part.

`I got 70.3% training accuracy as my max, 69.9% as the validation`



# Part 5: Weights and Biases

> Parts 5 and 6 are independent. Feel free to attempt them in any order you want.

> Background on W&B. W&B is a tool for tracking experiments. You can set up experiments and track metrics, hyperparameters, and even images. It's really neat and we highly recommend it. You can learn more about it [here](https://wandb.ai/site).
> 
> For this HW you have to use W&B. The next couple parts should be fairly easy if you setup logging for configs (hyperparameters) and for loss/accuracy. For a quick tutorial on how to use it, check out [this quickstart](https://docs.wandb.ai/quickstart). We will also cover it at HW party at some point this week if you need help.

## 5.0 Setup plotting for training and validation accuracy and loss curves. Plot a point every epoch.

`PUSH YOUR CODE TO YOUR OWN GITHUB :)`

## 5.1 Plot the training and validation accuracy and loss curves for AlexNet and LeNet. Attach the plot and any observations you have below.

`YOUR ANSWER HERE`

## 5.2 For just AlexNet, vary the learning rate by factors of 3ish or 10 (ie if it's 3e-4 also try 1e-4, 1e-3, 3e-3, etc) and plot all the loss plots on the same graph. What do you observe? What is the best learning rate? Try at least 4 different learning rates.

`YOUR ANSWER HERE`

## 5.3 Do the same with batch size, keeping learning rate and everything else fixed. Ideally the batch size should be a power of 2, but try some odd batch sizes as well. What do you observe? Record training times and loss/accuracy plots for each batch size (should be easy with W&B). Try at least 4 different batch sizes.

`YOUR ANSWER HERE`

## 5.4 As a followup to the previous question, we're going to explore the effect of batch size on _throughput_, which is the number of images/sec that our model can process. You can find this by taking the batch size and dividing by the time per epoch. Plot the throughput for batch sizes of powers of 2, i.e. 1, 2, 4, ..., until you reach CUDA OOM. What is the largest batch size you can support? What trends do you observe, and why might this be the case?
You only need to observe the training for ~ 5 epochs to average out the noise in training times; don't train to completion for this question! We're only asking about the time taken. If you're curious for a more in-depth explanation, feel free to read [this intro](https://horace.io/brrr_intro.html). 

`YOUR ANSWER HERE`

## 5.5 Try different data augmentations. Take a look [here](https://pytorch.org/vision/stable/transforms.html) for torchvision augmentations. Try at least 2 new augmentation schemes. Record loss/accuracy curves and best accuracies on validation/train set.

`YOUR ANSWER HERE`

## 5.6 (optional) Play around with more hyperparameters. I recommend playing around with the optimizer (Adam, SGD, RMSProp, etc), learning rate scheduler (constant, StepLR, ReduceLROnPlateau, etc), weight decay, dropout, activation functions (ReLU, Leaky ReLU, GELU, Swish, etc), etc.

`YOUR ANSWER HERE`



# Part 6: ResNet

## 6.0 Implement and train ResNet18

In `models/models.py`, we provided some skelly/guiding comments to implement ResNet. Implement it and train it on CIFAR10. Report training and validation curves, hyperparameters, best validation accuracy, and training time as compared to AlexNet. 

`YOUR ANSWER HERE`

## 6.1 Visualize examples

Visualize a couple of the predictions on the validation set (20 or so). Be sure to include the ground truth label and the predicted label. You can use `wandb.log()` to log images or also just save them to disc any way you think is easy.

`YOUR ANSWER HERE`


# Part 7: Kaggle submission

To make this more fun, we have scraped an entire new dataset for you! 🎉

We called it MediumImageNet. It contains 1.5M training images, and 190k images for validation and test each. There are 200 classes distributed approximately evenly. The images are available in 224x224 and 96x96 in hdf5 files. The test set labels are not provided :). 

The dataset is downloaded onto honeydew at `/data/medium-imagenet`. Feel free to play around with the files and learn more about the dataset.

For the kaggle competition, you need to train on the 1.5M training images and submit predictions on the 190k test images. You may validate on the validation set but you may not use is as a training set to get better accuracy (aka don't backprop on it). The test set labels are not provided. You can submit up to 10 times a day (hint: that's a lot). The competition ends on __TBD__.

Your Kaggle scores should approximately match your validation scores. If they do not, something is wrong.

(Soon) when you run the training script, it will output a file called `submission.csv`. This is the file you need to submit to Kaggle. You're required to submit at least once. 

## Kaggle writeup

We don't expect anything fancy here. Just a brief summary of what you did, what worked, what didn't, and what you learned. If you want to include any plots, feel free to do so. That's brownie points. Feel free to write it below or attach it in a separate file.

**REQUIREMENT**: Everyone in your group must be able to explain what you did! Even if one person carries (I know, it happens) everyone must still be able to explain what's going on!

Now go play with the models and have some competitive fun! 🎉

# Speech-Commands-Classification

Speech Commands Classification is a project that aims to classify the speech commands from the [Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) using a Convolutional Neural Network (CNN) and a Long Short-Term Memory (LSTM) network.
In this project we will use PyTorch to build the CNN to classify the speech commands.

## Installation
My recommendation is to install with requirements.txt file. 

```pip install -r requirements.txt```

Some of you probably have different cuda or gpu, so you can install [PyTorch](https://pytorch.org/get-started/locally/) with this tool.



## Dataset
You can download the new version from [here](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz).
Extract the dataset in the directory you wish to work in.
Split the dataset into train, validation and test sets using the following command:

```python main.py --action create_dataset --speech_commands_folder <google-command-folder> --out_path <path to save the splitted data>```

You can leave speech_commands_folder and out_path empty to use the default values.

To Transform the dataset from wav files to numpy array files you can use this command:

```python main.py --action transform_dataset --dataset <path-to-dataset-folder>```

You can leave dataset empty to use the default values.

## Architectures
The architectures of the CNN is the following:
* Standard [LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf).
* LeNet with several modifications. Which includes:
  * [Batch normalization](https://arxiv.org/abs/1502.03167).
  * [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
  * [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function.
* [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

  
## Training
If you want to train the CNN you can use the following command:

```python main.py --action train --dataset <path-to-dataset-folder> --model_name <'lenet', 'improved_lenet', 'alexnet'>```

There are several parameters you can use to train the CNN. You can see them by using the following command:

```python main.py --help```

## Testing
For testing the CNN you can use the following command:

```python main.py --action test --dataset <path-to-dataset> --model_checkpoint <path-to-model>``` 


## Results
| Model          | Test Accuracy | Test Weighted F1 |
|----------------|---------------|------------------|
| LeNet          | 81.4%         | 89.3%            | 
| Improved LeNet | 90%           | 94.6%            | 
| AlexNet        | 94%           | 96.8%            | 

You can see the validation accuracy and F1 for each epoch in the following figures:

Lenet in gray. Improved LeNet in blue. AlexNet in purple.

![Validation Accuracy](/figures/validation_accuracy.png)
![Validation F1](/figures/validation_weighted_f1.png)

# Speech-Commands-Classification

* Classifier for [Google command dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)
* The Classifier was implemented with PyTorch.
* I used [LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) architecture with some tweaks, like:
    * [Batch normalization](https://arxiv.org/abs/1502.03167)
    * [Kaiming initialization](https://arxiv.org/pdf/1502.01852v1.pdf)

## Features
* Training and testing LeNet.
* Arrange dataset in an Train, Test, Valid folders for easy loading.
* Transforming all the dataset to numpy array for easy batch loading in Google Colab.

## Installation
My recommendation is to install with requirements.txt file.
I removed pytorch from the requirements file, because every pc needs its configuration
* pip install -r requirements.txt
* Install [PyTorch](https://pytorch.org/get-started/locally/)

## Dataset
You can download the new version from [here](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz).

Don't forget to extract the dataset.

To Split the dataset i used [this code](https://github.com/adiyoss/GCommandsPytorch/blob/master/make_dataset.py), i added some improvements in this repo.

* ```python main.py create_dataset --gcommands_fold <google-command-folder> --out_path <path to save the data the new format>```

To Transform the dataset from wav files to numpy array files you can use this command:
* ```python main.py transform_dataset --dataset <path-to-dataset-folder>```

## Training
Just run:
 
* ```python main.py train --dataset <path-to-dataset-folder>```

## Testing
*  ```python main.py test --test_dataset <path-to-test-folder> --model <path-to-model>``` 


## Results
| Train acc. | Valid acc. | Test acc.|
| ------------- | ------------- | ------------- |
| 99%   | 89% | 88% | 

you can see them in the notebook

## Loss && Validation Accuracy Graph
![loss](/graphs/loss%20and%20accuracy.png)

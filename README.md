![banner](digit_recognition.png)

# Kaggle Digit Recognition

![last commit](https://img.shields.io/github/last-commit/Juanki0396/kaggle_DigitRecognition)
![your id](https://road-to-kaggle-grandmaster.vercel.app/api/simple/juanki0396)


This is my first GitHub repo where I will upload all my code related to the kaggle 
Machine Learning competition in [Digit recognition](https://www.kaggle.com/c/digit-recognizer).

The **main goal** is to create a model that makes good inference from the MNIST data. My **personal objective** is to put in practice my skills with different convolutional neural networks
learned in the previous months.

## Table of contents
- [Model info](#model-info)
- [Guide](#guide)
    - [Training](#training)
    - [Inference](#inference)
- [Future additions](#future-additions)

## Model info



## Guide

Usefull info about how the code works will be displayed here.

### Training

Training step is implmented in training.py. This scrpits comes with a light command line interface that allows you to set your own training. An example of how is it used is shown below:

    python training.py --path <train.csv path> --outputPath <dir to save the model>

Try to use -h to see all options

    python training.py -h

### Inference

Not yet implemented

## Future additions

- Dockerize inference with pretrained model
- Making visual inference app
- ...
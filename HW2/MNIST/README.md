# Ring loss in my easy network

## 1.Introduction

This repository includes the code for testing Ring loss function on my easy convolutional network. 

Ring loss function is introduced in [1]https://arxiv.org/abs/1803.00130  

the loss function in pytorch is implemented in https://github.com/Paralysis/ringloss

## 2.Installation

```
pip install -r requirements.txt
```



## 3.Training and Testing

Main.py automatically test the model in each training epoch.

If you want to use cpu:

```
python main.py --no-cuda True
```

to modify hyper-parameter, see parser in main.py

If you want some help:

```
python main.py help
```


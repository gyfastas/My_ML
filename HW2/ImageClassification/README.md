# MobilenetV2 On Dogs.vs Cats

## 1.Introduction

This repository includes the code for training and testing MobilenetV2 on Kaggle Dogs.vs Cats competition

MobileNetV2 is introduced  in [1]https://arxiv.org/abs/1801.04381

Generalized Training and testing code is from https://github.com/chenyuntc/pytorch-book which is really a good book for learning pytorch! This guy is amazing :)

## 2.Installation

```
pip install -r requirements.txt
```



## 3.Training and Testing

To train the model:

```
python Main.py train 
```

to modify hyper-parameter, see parser in config.py

To test the model:

```
python Main.py test --load_model_path ='Your trained model path'
```

This will return a result.csv, which you can upload on kaggle and get a score.

https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition 

Have fun!


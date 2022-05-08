# Interpretable Feedforward Neural Network (IFFNN)

On the Effectiveness of Interpretable Feedforward Neural Network (IJCNN 2022)

Publication:
M. Q. Li, B. C. M. Fung, and A. Abusitta. On the Effectiveness of Interpretable Feedforward Neural Network. In Proceedings of the International Conference on Joint Conference on Neural Networks (IJCNN), pages 1-8, Padova, Italy: IEEE, July 2022.   (Pages to be updated)



This repository contains the implementation of the IFFNNs proposed in our IJCNN 2022 paper and the evaluation code. 

Note: We are able to reproduce the exact results of fully-connected and highway neural networks (interpretable or non-interpretable) with our server we described in the paper: two Xeon E5-2697 CPUs, 384 GB of memory, and four Nvidia Titan XP graphics cards. However, even though we set the random seeds and use the same server, experiment results for convolutional neural networks and residual neural networks change every time. When we try to run the scripts on our other server with 4 RTX 2080Ti, all the results lightly changed, with <= 0.4% accuracy, except logistic regression and softmax regression. We use the same Python, Pytorch, Numpy, Scipy versions on both servers. Even though the results for the neural networks on the two machines are different, the results for logistic regression and softmax regression are always the same. That means the initialization of parameters is different even though we set the random seeds. 


#### Requirements
- Python 3.7.9
- Pytorch 1.6.0
- Scikit-learn	0.23.2
- Numpy 1.19.1
- Scipy 1.5.2
- Torchvision 0.7.0


### Usage

To run all experiments of neural networks including logistic regression and softmax regression on MNIST:
```shell
python evaluate_MNIST.py
```
To run all experiments of neural networks including logistic regression and softmax regression on INBEN:
```shell
python evaluate_INBEN.py
```
To run all experiments of decision tree on both datasets:
```shell
python evaluate_DT.py
```
To quantitatively evaluate the IFFNNs on INBEN, you need to run the evaluate_INBEN.py to train the models, and then run:
```shell
python evaluate_interpretability_INBEN.py
```

To qualitatively evaluate the IFFNNs on MNIST, you need to run the evaluate_MNIST.py to train the models, and then run:
```shell
python evaluate_interpretability_MNIST.py
```


Disclaimer:

The software is provided as-is with no warranty or support. We do not take any responsibility for any damage, loss of income, or any problems you might experience from using our software. If you have questions, you are encouraged to consult the paper and the source code. If you find our software useful, please cite our paper above.

Copyright 2022 McGill University. All rights reserved.

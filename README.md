# Interpretable Feedforward Neural Network (IFFNN)

On the Effectiveness of Interpretable Feedforward Neural Network (IJCNN 2022)

This repository contains the implementation of the IFFNNs proposed in our IJCNN 2022 paper and the evaluation code.


#### Requirements
- Python 3.7
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

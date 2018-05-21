# Water-Table-Depth-Prediction-Pytorch

### Introduction
This is a PyTorch implementation of our work *Developing a Long Short-Term Memory (LSTM) based Model for Predicting Water Table Depth in Agricultural Areas*.[[Paper](https://www.sciencedirect.com/science/article/pii/S0022169418303184)]

### Requirements
```
Python3.x
pytorch>=0.4.0
numpy>=1.14.0
pandas>=0.22.0
scikit-learn>=0.14
```
### Installation
The code was tested with Python 3.5. To use this code, please do:


0. Clone the repo:
    ```Shell
    git https://github.com/jfzhang95/Water-Table-Depth-Prediction-PyTorch
    cd Water-Table-Depth-Prediction-PyTorch
    ```
 
1. Install dependencies:
    ```Shell
    pip install matplotlib numpy pandas scikit-learn
    ```
    For pytorch installation, please see in [PyTorch.org](https://pytorch.org/).
  
2. To try the demo code, please run:
    ```Shell
    python demo.py
    ```

If installed correctly, the result should look like this:
![results](doc/results.png)

Noted that the demo data ([demo.csv](https://github.com/jfzhang95/Water-Table-Depth-Prediction-PyTorch/blob/master/data/demo.csv)) are processed manually,  so they are not real data, but they still can reflect the correlation between the original data.

### Citation
If you think our code is useful, please consider citing the following paper:

	@article{zjf18,
	  journal        = {Journal of Hydrology},
	  title          = {Developing a Long Short-Term Memory (LSTM) based Model for Predicting Water Table Depth in Agricultural Areas},
	  author         = {Jianfeng Zhang, Yan Zhu, Xiaoping Zhang, Ming Ye and Jinzhong Yang},
	  year           = {2018},
	  volume         = {561},
	  pages          = {918-929}
	}


### License
[MIT](https://github.com/jfzhang95/Water-Table-Depth-Prediction-PyTorch/blob/master/LICENSE)


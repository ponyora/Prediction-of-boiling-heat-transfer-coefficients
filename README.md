
# Prediction-of-boiling-heat-transfer-coefficients

# Installation

Install first Python.

Then install scikit-learn and GPy.  
https://scikit-learn.org/  
https://pypi.org/project/GPy/

Two frameworks are available for deep neural network; Keras and PyTorch.  
To use Keras version, install Keras.  
https://keras.io/  
To use PyTorch version, install PyTorch.  
https://pytorch.org/

# Usage

First, please prepare your own data of heat transfer coefficients. 
In this sample, the name of the data file is "input.csv" where the value of the first column are heat transfer coefficients, and 2-22th columns represent experimental conditions and physical properties.
Please change the file name and the number of columns so that this program matches your own data file.

Then, simply execute main_keras.py for Keras version or main_pytorch.py for PyTorch version.

This sample outputs the true values, the predicted values, and the corresponding variances to "output.csv".

# Author

* Yuichi Sei
* The University of Electro-Communications
* seiuny@uec.ac.jp


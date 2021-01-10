# TSUnify
A Unified Framework for Time Series Classification


Time series data is a very vast and easily available data source that researchers are
still trying to find optimal classification and prediction algorithms for. Similar to
image data, it is natural to want to use convolution networks and deep neural networks 
towards this task, however this has seen failure except for the last few years
with the launch of ROCKET and InceptionTime. 

Inspired by their contributions, this project endeavors to test various model 
architectures famously used in deep learning to this task after feature extraction 
using a relatively novel PPV feature vector first introduced in 2019, and thus frame a 
framework to develop future time series models on, agnostic to their primary task 
and data domain.


4 unique models were designed to test this feature vector, and also serve as a relative 
benchmark for future experimentation. An autoencoder architecture, a Dense-Net inspired 
architecture, a Res-Net inspired architecture and also a Inception-Net inspired 
architecture, were developed for this project. Supporting plots have been provided and 
the code is available for reference.



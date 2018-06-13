# Python-Potsdam
This project reads the parameters from the pre-trained network "Potsdam-denseprediction-distribution.mat".

It then converts the parameters in numpy format and initializes an equivalent CNN in tensorflow.

 - cnn_model.py contains the CNN architecture

 - the matfiles folder contains the Potsdam-denseprediction-distribution parameters from the matconvnet implementation

 - convert_mat_to_numpy.py contains a short script to write the weight matrixes as numpy vectors

 - the mat2numpy contains said numpy vectors in .npy format

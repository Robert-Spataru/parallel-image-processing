To see the speedup changes, run the following two commands (make sure that the Makefile is configured to your C++ compiler): 
1. make 
2. ./test
To modify code and run again, you must rebuild test each time you modify your parallel.c file.
To run a specific algorithm, specify by writing -p <n> after ./test where n is the algorithm number you are testing.


Algorithm 1: Mean Pixel Value - Implements a solution for calculating the mean pixel value in an image. 

Algorithm 2: Grayscale - Implements a solution for converting an image to grayscale. It also computes the maximum
grayscale value and the number of times it appears in the grayscale image. The grayscale value of a pixel
is calculated by taking the mean of its RGB values. Each of the RGB values in the grayscale image is set
to the grayscale value of the corresponding pixel from the original image.

Algorithm 3: Convolution - Implements a solution for computing a convolution between a kernel and a padded version of
the original image.
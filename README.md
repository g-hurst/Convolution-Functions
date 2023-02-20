# Convolution-Functions

### Overview:
Functions for perfroming convolution of an input image are in the file [convolution.c](https://github.com/g-hurst/Convolution-Functions/blob/main/convolution.c).

Testing of this function was done by comparing the output of the function `make_convolution` in convolution.c to the output of the [python convolve function from Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html). If the difference in each number between the outputs is less then 1%, the test was deemed to have passed.  
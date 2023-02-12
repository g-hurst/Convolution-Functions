#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

typedef struct _Layer {
 int m, n, c;
 float*** weights;
 
} Layer;

Layer* make_layer(int, int, int);
Layer* make_convolution(Layer* input, Layer* kernel);
void destroy_layer(Layer*);

#endif
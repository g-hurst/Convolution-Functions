#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

typedef struct _Layer {
 int m, n, c;
 double*** weights;
 
} Layer;

Layer* make_layer(int, int, int);
void make_convolution(Layer* input, Layer* kernel, Layer** final_out);
void destroy_layer(Layer*);

#endif
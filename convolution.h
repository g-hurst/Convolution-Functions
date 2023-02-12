#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

typedef struct _Layer {
 int m, n, c;
 float*** weights;
 
} Layer;

Layer* make_layer(int, int, int);
void destroy_layer(Layer*, int, int);

#endif
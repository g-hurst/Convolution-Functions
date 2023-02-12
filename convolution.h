#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

typedef struct _Layer {
 int m, n, c;
 float*** weights;
 
} Layer;

Layer* make_layer(int m, int n, int c);
void destroy_layer(Layer *layer, int m, int n, int c);

#endif
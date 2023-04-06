#include <stdlib.h>
#include <float.h>

#include "convolution.h"
#include "logging.h"

double get_weight(Layer* l, int c, int m, int n){
    if(!l)  return 0;
    int invalid = (m < 0 || n < 0 || c < 0) || (m > l->m || n > l->n || c > l->c);
    if(invalid) return 0;
    return l->weights[(c * l->m * l->n) + (m * l->n) + n];
}

void set_weight(double val, Layer* l, int c, int m, int n) {
    if(!l)  return;
    int invalid = (m < 0 || n < 0 || c < 0) || (m > l->m || n > l->n || c > l->c);
    if(invalid) return;
    l->weights[(c * l->m * l->n) + (m * l->n) + n] = val;
}

Layer* make_layer(int m, int n, int c){
    if( !m || !n || !c ) {
        return NULL;
    }

    Layer* layer = (Layer*) calloc(1, sizeof(*layer));
    *layer = (Layer) { .m=m, .n=n, .c=c, .weights=NULL };
    
    // creates a 3d matrix of zeros size (m, n, c)
    double* weights = (double*) calloc(m * n * c, sizeof(*weights));
    layer->weights = weights;
    for(int i=0; i < m; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < c; k++){
                set_weight(0, layer, k, i, j);
            }
        }
    }
    
    return layer;
}

static double dot_3d(Layer* input, Layer* kernel, int offest_c, int offset_i, int offset_j) {
    double product = 0;
    for (int i = 0; i < kernel->m; i++) {
        for(int j = 0; j < kernel->n; j++) {
            for(int k = 0; k < kernel->c; k++) {
                product += get_weight(input, k + offest_c, i + offset_i, j + offset_j) * get_weight(kernel, k, i, j);
            }
        }
    }
    return product;
}

static double get_max(Layer* input, int window_size_m, int window_size_n, int c, int offset_1, int offset_2) {
    double max = -DBL_MAX;
    double curr;
    for (int i = 0; i < window_size_m; i++) {
        for(int j = 0; j < window_size_n; j++) {
            curr = get_weight(input, c, i + offset_1, j + offset_2);
            if(max < curr) {
                max = curr;
            }
        }
    }
    return max;   
}

void make_max_pooling(Layer* input, int window_size_m, int window_size_n, int stride, Layer* output) {
    /*
    Takes an input layer, window shape, and stride. 
    Max pooling is then performed to create the final_out layer.
    NOTE: this assumes the window shape and stride will match the input 
          (this is NOT memory safe if it does not!!!!)
    */
    for(int i = 0; i < output->m; i++){
        for(int j = 0; j < output->n; j++) {
            for(int k = 0; k < output->c; k++) {
                double max = get_max(input, window_size_m, window_size_n, k, i * stride, j * stride);
                set_weight(max, output, k, i, j);
            }
        }
    }
}

void make_convolution(Layer* input, Layer* kernel, int padding, Layer* output){ 
    /*
    takes an input layer and a kernel and then convolutes it 
    into a output layer. 
    NOTE: this assumes the kernal to only have one channel
    */
    for(int i = -padding; i < output->m + padding; i++){
        for(int j = -padding; j < output->n + padding; j++) {
            for(int k = -padding; k < output->c + padding; k++) {
                double dot = dot_3d(input, kernel, k, i, j);
                set_weight(dot, output, k, i, j);
            }
        }
    }
}

void make_fully_connected(Layer* input, Layer* w_and_b, Layer* output){
    for (int i = 0; i < w_and_b->n; i += 2){
        for(int j = 0; j < input->m * input->n * input->c; j++){
            output->weights[i/2] += input->weights[j] * get_weight(w_and_b, 0, j, i);
            output->weights[i/2] += get_weight(w_and_b, 0, j, i+1);
        }
    }
}

void destroy_layer(Layer *layer){
    /*
    Takes a layer and frees all the mem required for it
    */
    if( !layer ) return;
    
    free(layer->weights);
    free(layer);
}
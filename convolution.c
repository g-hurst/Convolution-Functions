#include <stdlib.h>
#include <float.h>

#include "convolution.h"
#include "logging.h"

double get_weight(Layer* l, int c, int m, int n){
    return l->weights[(c * l->m * l->n) + (n * l->m) + m];
}

void set_weight(double val, Layer* l, int c, int m, int n) {
    l->weights[(c * l->m * l->n) + (n * l->m) + m] = val;
    // printf("setting weight (%d, %d, %d): %d\n", c, m, n, (c * l->m * l->n) + (n * l->m) + m);
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

static double dot_2d(Layer* input, Layer* kernel, int c, int offset_1, int offset_2) {
    double product = 0;
    for (int i = 0; i < kernel->m; i++) {
        for(int j = 0; j < kernel->n; j++) {
            product += get_weight(input, c, i + offset_1, j + offset_2) * get_weight(kernel, 0, i, j);
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

void make_max_pooling(Layer* input, int window_size_m, int window_size_n, int stride, Layer** final_out) {
    /*
    Takes an input layer, window shape, and stride. 
    Max pooling is then performed to create the final_out layer.
    NOTE: this assumes the window shape and stride will match the input 
          (this is NOT memory safe if it does not!!!!)
    */
    Layer* output = make_layer((input->m - window_size_m) / stride + 1, 
                               (input->n - window_size_n) / stride + 1,
                               input->c);

    for(int i = 0; i < output->m; i++){
        for(int j = 0; j < output->n; j++) {
            for(int k = 0; k < output->c; k++) {
                double max = get_max(input, window_size_m, window_size_n, k, i * stride, j * stride);
                set_weight(max, output, k, i, j);
            }
        }
    }

    *final_out =  output;
}

void make_convolution(Layer* input, Layer* kernel, Layer** final_out){ 
    /*
    takes an input layer and a kernel and then convolutes it 
    into a output layer. 
    NOTE: this assumes the kernal to only have one channel
    */
    Layer* output = make_layer(input->m - kernel->m + 1, 
                               input->n - kernel->n + 1,
                               input->c);

    for(int i = 0; i < output->m; i++){
        for(int j = 0; j < output->n; j++) {
            for(int k = 0; k < output->c; k++) {
                double dot = dot_2d(input, kernel, k, i, j);
                set_weight(dot, output, k, i, j);
            }
        }
    }
    *final_out =  output;
}

void destroy_layer(Layer *layer){
    /*
    Takes a layer and frees all the mem required for it
    */
    if( !layer ) return;
    
    free(layer->weights);
    free(layer);
}
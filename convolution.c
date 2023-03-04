#include <stdlib.h>

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
        // weights[i] = (double**) calloc(n, sizeof(**weights));
        for(int j = 0; j < n; j++){
            // weights[i][j] = (double*) malloc(c * sizeof(***weights));
            for(int k = 0; k < c; k++){
                set_weight(0, layer, k, i, j);
                // weights[i][j][k] = 0;
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
            // product += input->weights[i + offset_1][j + offset_2][c] * kernel->weights[i][j][0];
        }
    }
    return product;
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
                // output->weights[i][j][k] = dot_2d(input, kernel, k, i, j);
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

    // for (int i = 0; i < layer->m; i++){
    //     for (int j = 0; j < layer->n; j++){
    //         free(layer->weights[i][j]);
    //     }
    //     free(layer->weights[i]);
    // }
    free(layer->weights);
    free(layer);
}
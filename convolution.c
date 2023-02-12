#include <stdlib.h>

#include "convolution.h"
#include "logging.h"

Layer* make_layer(int m, int n, int c){
    if( !m || !n || !c ) {
        return NULL;
    }

    Layer* layer = calloc(1, sizeof(*layer));
    *layer = (Layer) { .m=m, .n=n, .c=c, .weights=NULL };
    
    // creates a 3d matrix of zeros size (m, n, c)
    float*** weights = calloc(m, sizeof(*weights));
    for(int i=0; i < m; i++){
        weights[i] = calloc(n, sizeof(**weights));
        for(int j = 0; j < n; j++){
            weights[i][j] = malloc(c * sizeof(***weights));
            for(int k = 0; k < c; k++){
                weights[i][j][k] = 0;
            }
        }
    }
    
    layer->weights = weights;
    return layer;
}

Layer* make_convolution(Layer* input, Layer* kernel){ 
    Layer* output = make_layer(input->m - kernel->m + 1, 
                               input->n - kernel->n + 1,
                               input->c);
    return output;

}

void destroy_layer(Layer *layer, int m, int n){
    /*
    Takes a layer and frees all the mem required for it
    */
    if( !layer ) return;

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            free(layer->weights[i][j]);
        }
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer);
}
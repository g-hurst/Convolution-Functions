#include <stdlib.h>

#include "convolution.h"

Layer* make_layer(int m, int n, int c){
    Layer* layer = malloc(sizeof(*layer));
    *layer = (Layer) { .m=m, .n=n, .c=c, .weights=NULL };
    layer->weights = calloc(m, sizeof(float));
    for(int i=0; i < m; i++){
        layer->weights[m] = calloc(n, sizeof(float));
        for(int j = 0; j < n; j++){
            layer->weights[m][n] = calloc(c, sizeof(float));
            for(int k = 0; k < c; k++){
                layer->weights[m][n][k] = 0;
            }
        }
    }

}

void destroy_layer(Layer *layer, int m, int n, int c){
    /*
    Takes a layer and frees all the mem required for it
    */
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            free(layer->weights[i][j]);
        }
        free(layer->weights[i]);
    }
    free(layer->weights);
}
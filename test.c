#include <stdio.h>
#include "convolution.h"
#include "logging.h"

#define MY_RAND_MIN -10
#define MY_RAND_MAX  10

static void assign_random(Layer* l);

int main(){

    Layer* input_layer = make_layer(10, 10, 1);
    Layer* kernel      = make_layer(3, 3, 1);

    assign_random(input_layer);
    assign_random(kernel);

    DBG_PRINTF("Input Layer:\n");
    DBG_PRINT_LAYER(*input_layer, 0);
    DBG_PRINTF("Kernel:\n");
    DBG_PRINT_LAYER(*kernel, 0);
    
    Layer* conv = make_convolution(input_layer, kernel);
    DBG_PRINTF("Conv ayer:\n");
    DBG_PRINT_LAYER(*conv, 0);

    // free mem
    destroy_layer(input_layer);
    destroy_layer(kernel);
    destroy_layer(conv);
    

    return 0;
}

static void assign_random(Layer* l) {
    for (int i = 0; i < l->m; i++){
        for(int j = 0; j < l->n; j++){
            for (int k = 0; k < l->c; k++){
                l->weights[i][j][k] = rand() % (MY_RAND_MAX - MY_RAND_MIN + 1) + MY_RAND_MIN;
            }
        }
    }
}
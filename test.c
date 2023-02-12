#include <stdio.h>
#include "convolution.h"
#include "logging.h"


int main(){
    DBG_PRINTF("Running...\n");
    Layer* layer = make_layer(5, 5, 5);

    DBG_PRINTF("Created layer\n");
    DBG_PRINT_LAYER(*layer, 0);
    destroy_layer(layer, 5, 5);
    

    return 0;
}
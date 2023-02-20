#include <stdio.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <iostream>
#include <math.h>

#include "convolution.h"
#include "logging.h"

#define TOTAL_TESTS 6

static bool run_test(const char* f_name);

int main(){
    system("python3 generate_tests.py"); // runs the py file to generate the test cases

    // test cases
    const char* f_names[TOTAL_TESTS] = {"keys/test_0.json", "keys/test_1.json", "keys/test_2.json",
                                        "keys/test_3.json", "keys/test_4.json", "keys/test_5.json"};

    for (int i = 0; i < TOTAL_TESTS; i++) {
        if( run_test(f_names[i])) {
            LOG_GREEN("Test %02d Passed: %s\n", i, f_names[i]);
        }
        else{
            LOG_RED("Test %02d Failed: %s\n", i, f_names[i]);
        }
    }
    return 0;
}

static Layer* json_to_layer(Json::Value mat){
    int m = mat["shape"][0].asInt();
    int n = mat["shape"][1].asInt();
    Layer* layer = make_layer(m, n, 1);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            layer->weights[i][j][0] = mat["data"][i][j].asDouble();
        }
    }
    return layer;
}

static bool run_test(const char* f_name){
    Json::Value data;
    std::ifstream data_file(f_name, std::ifstream::binary);
    data_file >> data;   
    
    bool is_same = true;

    Layer* mat = json_to_layer(data["matrix"]);
    Layer* ker = json_to_layer(data["kernel"]);
    Layer* conv_key  = json_to_layer(data["convolution"]);
    Layer* conv_calc = make_convolution(mat, ker);
    
    DBG_PRINT_LAYER(*mat, 0);
    DBG_PRINTF("\n");
    DBG_PRINT_LAYER(*ker, 0);
    DBG_PRINTF("\nkey convolution: \n");
    DBG_PRINT_LAYER(*conv_key, 0);
    DBG_PRINTF("calculated convolution: \n");
    DBG_PRINT_LAYER(*conv_calc, 0);

    for(int i = 0; i < conv_key->m; i++){
        for(int j = 0; j < conv_key->n; j++){
            double diff = fabs(conv_key->weights[i][j][0] - conv_calc->weights[i][j][0]) / fabs(conv_key->weights[i][j][0]);
            is_same &= (diff < 0.01);
        }
    }

    destroy_layer(mat);
    destroy_layer(ker);
    destroy_layer(conv_key);
    destroy_layer(conv_calc);

    return is_same;
}

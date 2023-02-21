#include <stdio.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <iostream>
#include <math.h>

#include "convolution.h"
#include "logging.h"

#define TESTS_2D 6
#define TESTS_3D 6

static bool run_test(const char* f_name);

int main(){
    system("python3 generate_tests.py"); // runs the py file to generate the test cases

    // test cases for 2d convolution
    const char* f_names_2d[TESTS_2D] = {"keys/test_0.json", "keys/test_1.json", "keys/test_2.json",
                                        "keys/test_3.json", "keys/test_4.json", "keys/test_5.json"};
    LOG_BLUE("Running test cases for 2d convolution: \n");
    for (int i = 0; i < TESTS_2D; i++) {
        if( run_test(f_names_2d[i])) {
            LOG_GREEN("Test %02d Passed: %s\n", i, f_names_2d[i]);
        }
        else{
            LOG_RED("Test %02d Failed: %s\n", i, f_names_2d[i]);
        }
    }

    // test cases for 3d convolution
    LOG_BLUE("Running test cases for 3d convolution (3 channel input): \n");
    const char* f_names_3d[TESTS_3D] = {"keys/test_6.json", "keys/test_7.json", "keys/test_8.json",
                                        "keys/test_9.json", "keys/test_10.json", "keys/test_11.json"};
    for (int i = 0; i < TESTS_3D; i++) {
        if( run_test(f_names_3d[i])) {
            LOG_GREEN("Test %02d Passed: %s\n", i, f_names_3d[i]);
        }
        else{
            LOG_RED("Test %02d Failed: %s\n", i, f_names_3d[i]);
        }
    }
    return 0;
}

static Layer* json_to_layer(Json::Value mat, int channel){
    int m = mat["shape"][1].asInt();
    int n = mat["shape"][2].asInt();
    Layer* layer = make_layer(m, n, 1);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            layer->weights[i][j][0] = mat["data"][channel][i][j].asDouble();
        }
    }
    return layer;
}

static bool run_test(const char* f_name){
    Json::Value data;
    std::ifstream data_file(f_name, std::ifstream::binary);
    data_file >> data;   
    
    bool is_same = true;
    int channels = data["matrix"]["shape"][0].asInt();
    Layer* ker = json_to_layer(data["kernel"], 0);

    for(int chan = 0; chan < channels; chan++) {
        Layer* mat = json_to_layer(data["matrix"], chan);
        Layer* conv_key  = json_to_layer(data["convolution"], chan);
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
        destroy_layer(conv_key);
        destroy_layer(conv_calc);
    }
    destroy_layer(ker);

    return is_same;
}

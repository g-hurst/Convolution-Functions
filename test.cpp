#include <stdio.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <iostream>
#include <math.h>

#include "convolution.h"
#include "logging.h"

#define TESTS_2D 6
#define TESTS_3D 6

#define MAX_ERR 0.01 // value for the max error between any expected and calcualted convolution value

static bool run_test(const char* f_name);

int main(){
    system("python3 generate_tests.py"); // runs the py file to generate the test cases

    // test cases for 2d convolution
    const char* f_names_2d[] = {"keys/test_0.json", "keys/test_1.json", "keys/test_2.json",
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
    const char* f_names_3d[] = {"keys/test_6.json", "keys/test_7.json", "keys/test_8.json",
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

static Layer* json_to_layer(Json::Value mat){
    int c = mat["shape"][0].asInt();
    int m = mat["shape"][1].asInt();
    int n = mat["shape"][2].asInt();

    // write each value fron the JSON obj to the Layer struct
    // printf("(c, m, n) == (%d, %d, %d)\n", c, m, n);
    Layer* layer = make_layer(m, n, c);
    for(int chan = 0; chan < c; chan++){
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                set_weight(mat["data"][chan][i][j].asDouble(), layer, chan, i, j);
                // layer->weights[i][j][chan] = mat["data"][chan][i][j].asDouble();
            }
        }
    }
    return layer;
}

static bool run_test(const char* f_name){
    // reads JSON file and stores it
    Json::Value data;
    std::ifstream data_file(f_name, std::ifstream::binary);
    data_file >> data;   
    data_file.close();

    // read from JSON and convert to layer structure
    Layer* mat = json_to_layer(data["matrix"]);
    DBG_PRINT_LAYER(mat, 0);

    Layer* ker = json_to_layer(data["kernel"]);
    DBG_PRINT_LAYER(ker, 0);

    Layer* conv_key  = json_to_layer(data["convolution"]);
    Layer* conv_calc;
    
    make_convolution(mat, ker, &conv_calc);
    
    // output for debugging when -D DEBUGGING is compiled
    
    DBG_PRINTF("\n");
    DBG_PRINTF("\nkey convolution: \n");
    DBG_PRINT_LAYER(conv_key, 1);
    DBG_PRINTF("calculated convolution: \n");
    DBG_PRINT_LAYER(conv_calc, 0);
    DBG_PRINT_LAYER(conv_calc, 1);
    DBG_PRINT_LAYER(conv_calc, 2);

    // finds the % difference between all the expected and calculated values
    // if any of them are > MAX_ERR, function reutrns false
    bool is_same = true;
    for(int chan = 0; is_same && chan < conv_key->c; chan++){
        for(int i = 0; is_same && i < conv_key->m; i++){
            for(int j = 0; is_same && j < conv_key->n; j++){
                // double diff = fabs(conv_key->weights[i][j][chan] - conv_calc->weights[i][j][chan]) / fabs(conv_key->weights[i][j][chan]);
                double w1 = get_weight(conv_key, chan, i, j);
                double w2 = get_weight(conv_calc, chan, i, j);
                double diff = fabs(w1 - w2) / fabs(w1);
                is_same &= (diff < MAX_ERR);
            }
        }
    }
    
    // mem cleanup
    destroy_layer(ker);
    destroy_layer(mat);
    destroy_layer(conv_key);
    destroy_layer(conv_calc);

    return is_same;
}

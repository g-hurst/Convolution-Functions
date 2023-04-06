#include <stdio.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <iostream>
#include <math.h>

#include "convolution.h"
#include "logging.h"

#define TESTS_2D 6
#define TESTS_3D 6
#define TESTS_MAXPOOL 2
#define TESTS_FULLCONN 2

#define MAX_ERR 0.01 // value for the max error between any expected and calcualted convolution value

static bool run_test_conv(const char* f_name, const int padding);
static bool run_test_maxpool(const char* f_name);
static bool run_test_fullconn(const char* f_name);

int main(){
    system("python3 generate_tests.py"); // runs the py file to generate the test cases

    // test cases for 2d convolution
    const char* f_names_2d[] = {"keys/test_conv_0.json", "keys/test_conv_1.json", "keys/test_conv_2.json",
                                        "keys/test_conv_3.json", "keys/test_conv_4.json", "keys/test_conv_5.json"};
    LOG_BLUE("Running test cases for 2d convolution: \n");
    for (int i = 0; i < TESTS_2D; i++) {
        if( run_test_conv(f_names_2d[i], 0)) {
            LOG_GREEN("Test %02d Passed: %s\n", i, f_names_2d[i]);
        }
        else{
            LOG_RED("Test %02d Failed: %s\n", i, f_names_2d[i]);
        }
    }

    // test cases for 3d convolution
    LOG_BLUE("Running test cases for 3d convolution: \n");
    const char* f_names_3d[] = {"keys/test_conv3D_0.json", "keys/test_conv3D_1.json", "keys/test_conv3D_2.json",
                                        "keys/test_conv3D_3.json", "keys/test_conv3D_4.json", "keys/test_conv3D_5.json"};
    for (int i = 0; i < TESTS_3D; i++) {
        bool success = (i < 6) ? run_test_conv(f_names_3d[i], 0) : run_test_conv(f_names_3d[i], 1);
        if( success ) {
            LOG_GREEN("Test %02d Passed: %s\n", i, f_names_3d[i]);
        }
        else{
            LOG_RED("Test %02d Failed: %s\n", i, f_names_3d[i]);
        }
    }

    // test cases for max pooling
    LOG_BLUE("Running test cases for max pooling: \n");
    const char* f_names_maxpool[] = {"keys/test_maxpool_0.json", "keys/test_maxpool_alexnet_1.json", "keys/test_maxpool_alexnet_3.json"};

    for (int i = 0; i < TESTS_MAXPOOL; i++) {
        if( run_test_maxpool(f_names_maxpool[i])) {
            LOG_GREEN("Test %02d Passed: %s\n", i, f_names_maxpool[i]);
        }
        else{
            LOG_RED("Test %02d Failed: %s\n", i, f_names_maxpool[i]);
        }
    }

    // test cases for fully connected layer
    LOG_BLUE("Running test cases for fully connected: \n");
    const char* f_names_fullconn[] = {"keys/test_fullconn_0.json", "keys/test_fullconn_1.json"};

    for (int i = 0; i < TESTS_FULLCONN; i++) {
        if( run_test_fullconn(f_names_fullconn[i])) {
            LOG_GREEN("Test %02d Passed: %s\n", i, f_names_fullconn[i]);
        }
        else{
            LOG_RED("Test %02d Failed: %s\n", i, f_names_fullconn[i]);
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

static bool is_valid_err(Layer* a, Layer* b, double err){
    // finds the % difference between all the expected and calculated values
    // if any of them are > MAX_ERR, function reutrns false
    double w1, w2, diff;
    bool is_same = (a->m == b->m) && (a->n == b->n) && (a->c == b->c);
    for(int chan = 0; is_same && chan < a->c; chan++){
        for(int i = 0; is_same && i < a->m; i++){
            for(int j = 0; is_same && j < a->n; j++){
                w1 = get_weight(a, chan, i, j);
                w2 = get_weight(b, chan, i, j);
                diff = fabs(w1 - w2) / fabs(w1);
                is_same &= (diff < err);
            }
        }
    }
    return is_same;
}

static bool run_test_fullconn(const char* f_name){
    // reads JSON file and stores it
    Json::Value data;
    std::ifstream data_file(f_name, std::ifstream::binary);
    data_file >> data;   
    data_file.close();

    // read from JSON and convert to layer structure
    Layer* mat = json_to_layer(data["matrix"]);
    DBG_PRINT_LAYER(mat, 0);
    DBG_PRINT_ARR(mat->weights, mat->m * mat->n * mat->c);

    Layer* w_and_b = json_to_layer(data["w_and_b"]);
    DBG_PRINT_LAYER(w_and_b, 0);

    Layer* fullconn_key  = json_to_layer(data["output"]);
    Layer* fullconn_calc = make_layer(w_and_b->n / 2, 1, 1);;
    
    make_fully_connected(mat, w_and_b, fullconn_calc);
    
    DBG_PRINTF("\n");
    DBG_PRINTF("\nkey fully connected: \n");
    DBG_PRINT_LAYER(fullconn_key, 0);
    DBG_PRINTF("calculated fully connected: \n");
    DBG_PRINT_LAYER(fullconn_calc, 0);

    // calculate the error between the calculation and the key
    bool is_same = is_valid_err(fullconn_key, fullconn_calc, MAX_ERR);
    
    // mem cleanup
    destroy_layer(w_and_b);
    destroy_layer(mat);
    destroy_layer(fullconn_key);
    destroy_layer(fullconn_calc);

    return is_same;
}

static bool run_test_maxpool(const char* f_name){
    // reads JSON file and stores it
    Json::Value data;
    std::ifstream data_file(f_name, std::ifstream::binary);
    data_file >> data;   
    data_file.close();

    // read from JSON and convert to layer structure
    Layer* mat = json_to_layer(data["matrix"]);
    DBG_PRINT_LAYER(mat, 0);

    int window_size_m = data["pool"]["shape"][0].asInt();
    int window_size_n = data["pool"]["shape"][0].asInt();
    int stride        = data["pool"]["stride"].asInt();

    Layer* pool_key  = json_to_layer(data["output"]);
    Layer* pool_calc = make_layer((mat->m - window_size_m) / stride + 1, 
                                    (mat->n - window_size_n) / stride + 1,
                                    mat->c);;
    
    make_max_pooling(mat, window_size_m, window_size_n, stride, pool_calc);
    
    DBG_PRINTF("\n");
    DBG_PRINTF("\nkey pool: \n");
    DBG_PRINT_LAYER(pool_key, 0);
    DBG_PRINTF("calculated pool: \n");
    DBG_PRINT_LAYER(pool_calc, 0);

    // calculate the error between the calculation and the key
    bool is_same = is_valid_err(pool_key, pool_calc, MAX_ERR);
    
    // mem cleanup
    destroy_layer(mat);
    destroy_layer(pool_key);
    destroy_layer(pool_calc);

    return is_same;
}

static bool run_test_conv(const char* f_name, const int padding){
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
    Layer* conv_calc = make_layer(mat->m - ker->m + 1 + 2 * padding, 
                                mat->n - ker->n + 1 + 2 * padding,
                                mat->c);;

    make_convolution(mat, ker, padding, conv_calc);
    
    DBG_PRINTF("\n");
    DBG_PRINTF("\nkey convolution: \n");
    DBG_PRINT_LAYER(conv_key, 0);
    DBG_PRINTF("calculated convolution: \n");
    DBG_PRINT_LAYER(conv_calc, 0);

    // calculate the error between the calculation and the key
    bool is_same = is_valid_err(conv_key, conv_calc, MAX_ERR);
    
    // mem cleanup
    destroy_layer(ker);
    destroy_layer(mat);
    destroy_layer(conv_key);
    destroy_layer(conv_calc);

    return is_same;
}

#ifndef __LOGGING_H__
    #define __LOGGING_H__
    #include <stdio.h>
    #include <stdlib.h>
    #include <stdbool.h>
    #include <unistd.h>
    
    //_______________________________FUNCTIONAL_TOOLS__________________________
    #define MAX(a,b) ((a) > (b) ? a : b)

    //_______________________________LOGGING_TOOLS__________________________
    #define ANSI_RED   "\x1b[31m" 
    #define ANSI_RESET "\x1b[0m"
    #define LOG_OUT stderr
    #define safe_write(sf_ptr, sf_sz, sf_num, sf_fp)           \
        do {                                                   \
            if(sf_fp)                                          \
                fwrite((sf_ptr), (sf_sz), (sf_num), (sf_fp));  \
        } while(false)  

    #define LOG_IS_FILE_OUT isatty(STDOUT_FILENO) 
    #define _LOG_OUT(...) fprintf(LOG_OUT, __VA_ARGS__)
    #define LOG_ERR(...)                            \
            do{                                     \
                if (LOG_IS_FILE_OUT) {              \
                    _LOG_OUT(ANSI_RED __VA_ARGS__); \
                    _LOG_OUT(ANSI_RESET);	        \
                }                                   \
                else{                               \
                    _LOG_OUT(__VA_ARGS__);          \
                }                                   \
            }while(false)

    //_______________________________DEBUGGING__________________________
    #ifdef DEBUGGING
        #define DBG_PRINTF(...) fprintf(stdout, __VA_ARGS__) 
        #define DBG_PRINT_ARR(dbg_arr, dbg_sz)                  \
        do {                                                    \
            DBG_PRINTF("[");                                    \
            for(int dbg_i = 0; dbg_i < (dbg_sz) - 1; dbg_i++){  \
                DBG_PRINTF("%3d, ", (dbg_arr)[dbg_i]);           \
            }                                                   \
            DBG_PRINTF("%3d]\n", (dbg_arr)[(dbg_sz) - 1]);       \
        } while(false)
        
        #define DBG_PRINT_ARR_C(dbg_arr, dbg_sz)                \
        do {                                                    \
            DBG_PRINTF("[");                                    \
            for(int dbg_i = 0; dbg_i < (dbg_sz) - 1; dbg_i++){  \
                DBG_PRINTF("%3c, ", (dbg_arr)[dbg_i]);           \
            }                                                   \
            DBG_PRINTF("%3c]\n", (dbg_arr)[(dbg_sz) - 1]);       \
        } while(false)

        #define DBG_PRINT_GRID(dbg_grid)                             \
        do {                                                         \
            for(int dbg_j = 0; dbg_j < (dbg_grid).m; dbg_j++){       \
                DBG_PRINT_ARR((dbg_grid).grid[dbg_j], (dbg_grid).n); \
            }                                                        \
        } while(false)
    #else
        #define DBG_PRINTF(...)
        #define DBG_PRINT_ARR(dbg_arr, dbg_sz)
        #define DBG_PRINT_ARR_C(dbg_arr, dbg_sz)
        #define DBG_PRINT_GRID(dbg_grid)
    #endif
#endif
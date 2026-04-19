#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cereal.h"
#include "array_matrix_funcs.h"

#define I_SIZE 784
#define H0_SIZE 16
#define H1_SIZE 16
#define L_SIZE 10

#define INIT_LOW 0
#define INIT_HIGH 10


int main() {

    // INPUT ARRAY
    array_ptr input = new_array(I_SIZE);
    
    // H0 ARRAY, WEIGHTS, and BIASES
    array_ptr H0 = new_array(H0_SIZE);
    matrix_ptr H0_W = new_matrix(H0_SIZE, I_SIZE);
    array_ptr H0_B = new_array(H0_SIZE);

    // H1 ARRAY
    array_ptr H1 = new_array(H1_SIZE);
    matrix_ptr H1_W = new_matrix(H1_SIZE, H0_SIZE);
    array_ptr H1_B = new_array(H1_SIZE);

    // OUTPUT ARRAY
    array_ptr output = new_array(L_SIZE);
    matrix_ptr L_W = new_matrix(L_SIZE, H1_SIZE);
    array_ptr L_B = new_array(L_SIZE);


    return 0;
}






// MVM CODE (TO BE OPTIMIZED)

void matrix_vector_mult() {

}
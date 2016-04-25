#ifndef CARS_CALCULATE_H
#define CARS_CALCULATE_H
#include "cars_type.h"
#include <iostream>


namespace cars
{

    // return A * B
    double multiply(cars_vector & A, cars_vector & B);

    // C = __A * B
    bool multiply(cars_tensor & A, cars_vector & B, cars_matrix & C);

    // return D = A * __B * C
    bool multiply(cars_vector & A, cars_tensor & B, cars_vector & C, cars_vector & D);

    // multiply by reversed order
    // return A * B * __C * D
    double multiply(cars_vector & A, cars_vector & B, cars_tensor & C, cars_vector & D);

    // D = __A * B * C
    bool multiply(cars_tensor & A, cars_vector & B, cars_vector & C, cars_vector & D);

    // B = `A
    bool transpose(cars_tensor &A, cars_tensor &B);

    bool initializeVector(cars_vector & A);

    bool printVector(cars_vector & A, std::ostream & fout);
    bool printMatrix(cars_matrix & A, std::ostream & fout);
    bool printTensor(cars_tensor & A, std::ostream & fout);


    bool replace(cars_vector & dest, cars_vector & src);
    bool replace(cars_tensor & dest, cars_tensor & src);
            

}

#endif

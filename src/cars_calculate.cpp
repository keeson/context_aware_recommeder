#include "cars_type.h"
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <omp.h>

namespace cars
{
    using namespace std;
    bool initializeVector(cars_vector & A)
    {
        #pragma omp parallel for
        for (int i = 0; i < A.size(); ++i){
            A[i] = 0.01 - 0.02 * rand() / RAND_MAX;
        }
        return true;
    }

    bool printVector(cars_vector & A, ostream & fout)
    {
        for (int i = 0; i < A.size(); ++i){
            fout << A[i];
            if (i + 1 < A.size()){
                fout << "#";
            }
        }
        return true;
    }

    bool printMatrix(cars_matrix & A, ostream & fout)
    {
        for (int i = 0; i < A.size(); ++i){
            printVector(A[i], fout);
            if (i + 1 < A.size()){
                fout << ",";
            }
        }
        return true;
    }

    bool printTensor(cars_tensor & A, ostream & fout)
    {
        for (int i = 0; i < A.size(); ++i){
            printMatrix(A[i], fout);
            if (i + 1 < A.size()){
                fout << ";";
            }            
        }
        return true;
    }


    // return A * B
    double multiply(cars_vector & A, cars_vector & B)
    {
        assert(A.size() == B.size());
        double res = 0.0;
        for (int i = 0; i < A.size(); ++i){
            res += A[i] * B[i];
        }
        return res;
    }

    // C = __A * B
    bool multiply(cars_tensor & A, cars_vector & B, cars_matrix & C)
    {
        C.resize(A.size());
        #pragma omp parallel for
        for (int i = 0; i < C.size(); ++i){
            C[i].resize(A[i].size());            
        }
//        printVector(B);
        assert(B.size() == A[0][0].size());
        #pragma omp parallel for
        for (int i = 0; i < A.size(); ++i){
            #pragma omp parallel for
            for (int j = 0; j < A[i].size(); ++j){
                C[i][j] = 0.0;
                for (int k = 0; k < A[i][j].size(); ++k){
                    C[i][j] += A[i][j][k] * B[k];
                }
            }
        }
        return true;
    }

    // return D = A * __B * C
    bool multiply(cars_vector & A, cars_tensor & B, cars_vector & C, cars_vector & D)
    {
        cars_matrix E;
        multiply(B, C, E);
        D.resize(E[0].size());
        assert(E.size() == A.size());
        #pragma omp parallel for
        for (int j = 0; j < D.size(); ++j){
            D[j] = 0.0;
            for (int i = 0; i < E.size(); ++i){
                D[j] += E[i][j] * A[i];
            }
        }
        return true;
    }

    // multiply by reversed order
    // return A * B * __C * D
    double multiply(cars_vector & A, cars_vector & B, cars_tensor & C, cars_vector & D)
    {
        cars_vector E;
        multiply(B, C, D, E);
//        printVector(E);
        return multiply(A, E);
    }

    // D = __A * B * C
    bool multiply(cars_tensor & A, cars_vector & B, cars_vector & C, cars_vector & D)
    {
        cars_matrix E;
        multiply(A, B, E);
        D.resize(E.size());
        assert(E[0].size() == C.size());
        #pragma omp parallel for
        for (int i = 0; i < E.size(); ++i){
            D[i] = 0.0;
            for (int j = 0; j < E[i].size(); ++j){
                D[i] += E[i][j] * C[j];
            }
        }
        return true;
    }

    // B = `A
    bool transpose(cars_tensor &A, cars_tensor &B)
    {
        B.resize(A[0][0].size());
        #pragma omp parallel for
        for (int i = 0; i < B.size(); ++i){
            B[i].resize(A[0].size());
            #pragma omp parallel for
            for (int j = 0; j < B[i].size(); ++j){
                B[i][j].resize(A.size());
                #pragma omp parallel for
                for (int k = 0;  k < B[i][j].size(); ++k){
                    B[i][j][k] = A[k][j][i];
                }
            }
        }
    }

    bool replace(cars_vector & dest, cars_vector & src)
    {
        assert(dest.size() == src.size());
        #pragma omp parallel for
        for (int i = 0; i < dest.size(); ++i){
            dest[i] = src[i];
        }
    }
    bool replace(cars_tensor & dest, cars_tensor & src){
        assert(dest.size() == src.size() && dest[0].size() == src[0].size() && dest[0][0].size() == src[0][0].size());
        #pragma omp parallel for
        for (int i = 0; i < dest.size(); ++i){
            #pragma omp parallel for
            for (int j = 0; j < dest[i].size(); ++j){
                #pragma omp parallel for
                for(int k = 0; k <dest[i][j].size(); ++k){
                    dest[i][j][k] = src[i][j][k];
                }
            }
        }
    }
}


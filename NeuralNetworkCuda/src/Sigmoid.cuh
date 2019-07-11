#pragma once 
#include "Utils.cuh"


class Sigmoid {
public:
    Sigmoid(const Sigmoid& other) {printf("sigmoid copy"); }

    __device__ static val_t activate(const val_t& val){
        return 1.0 / (1.0 + expf(-val));
    }

    __device__ static val_t derivative(const val_t& val) {
        return val * (1 - val);
    }
};
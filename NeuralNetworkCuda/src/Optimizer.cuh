#pragma once
#include "Utils.cuh"

struct optimizer {
    optimizer() = default;
    optimizer(const optimizer& other) = default;
    ~optimizer() = default;
    virtual void update(const val_t* deltas, val_t* weights, size_t weightsSize, cudaStream_t stream) = 0;
    virtual void reset() {} // override to implement pre-learning action
};




struct gradient_descent : public optimizer {
public:
    gradient_descent(val_t alpha, val_t lambda) : alpha(alpha), lambda(lambda) {}
    gradient_descent() : gradient_descent(0.01,0) {}

    void update(const val_t* deltas, val_t* weights, size_t weightsSize, cudaStream_t stream)override;
private:
    val_t alpha; // learning rate
    val_t lambda; // weight decay
};
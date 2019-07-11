#pragma once
#include "Utils.cuh"

class LossFunction {
public:
    virtual val_t f(const val_t* output, const val_t* expectedOutput, const size_t size, cudaStream_t stream = 0) = 0;
    virtual void df( const val_t* output,  const val_t* expectedOutput,
          val_t* errorGradientsBuffer, const size_t size, cudaStream_t stream = 0) = 0;
};

class MSE : public LossFunction {
public:
    MSE();
    val_t f( const val_t* output, const val_t* expectedOutput, const size_t size, cudaStream_t stream)override;
    void df( const val_t* output,  const val_t* expectedOutput,
         val_t* errorGradientsBuffer, const size_t size, cudaStream_t stream)override;

private:
    val_t* d_val;
    val_t h_val;
};

class NopLoss : public LossFunction{
    virtual val_t f(const val_t* output, const val_t* expectedOutput, const size_t size, cudaStream_t stream)override;
    virtual void df( const val_t* output,  const val_t* expectedOutput,
     val_t* errorGradientsBuffer, const size_t size, cudaStream_t stream)override;
};
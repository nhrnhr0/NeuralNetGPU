#pragma once
#include "Layer.cuh"
#include "math.h"
#include "Utils.cuh"
#include <algorithm>

template<class activationFunction>
__global__ void ActivationLayer_FF_Kernal(const val_t* inputs,val_t* outputs, size_t size) {
    for(size_t index = (blockIdx.x * blockDim.x + threadIdx.x); index < size; index += gridDim.x*blockDim.x) {
        outputs[index] = activationFunction::activate(inputs[index]);
    }
}

template<class activationFunction>
__global__ void ActivationLayer_BP_Kernal(
    const val_t* outputs, const val_t* errorGradients,
     val_t* outErrorGradients, size_t size) {
        for(size_t index = (blockIdx.x * blockDim.x + threadIdx.x); index < size; index += gridDim.x*blockDim.x) {
            outErrorGradients[index] = errorGradients[index] * activationFunction::derivative(outputs[index]);
        }
}

template<class activationFunction> // activationFunction: sigmoid/relu/...
class ActivationLayer : public Layer {

public:
    ActivationLayer();
    void Init(const Shape3d& inputShape) override;
    void FeedForward()override;
    void Backpropagate()override;

protected:
private:
    int threads, blocks;
};

template<class activationFunction>
ActivationLayer<activationFunction>::ActivationLayer(): Layer() {

}

template<class activationFunction>
void ActivationLayer<activationFunction>::Init(const Shape3d& inputShape) {
    setInputShape(inputShape);
    setOutputShape(inputShape);
    devAlloc((void**)&output, sizeof(val_t)*inputShape.size());
    devAlloc((void**)&outErrorGradients, sizeof(val_t)*inputShape.size());
    
    threads = DEFAULT_THREAD_SIZE;
    blocks = std::min((int)ceilf(inputShape.size()/(float)threads), DEFAULT_BLOCK_SIZE);
}




template<class activationFunction>
void ActivationLayer<activationFunction>::FeedForward() {
    //ActivationLayer_FF_Kernal<activationFunction><<<blocks, threads,0, streamProvider->get(0)>>>
    //    (getInput(), output, getInputShape().size());
}



template<class activationFunction>
void ActivationLayer<activationFunction>::Backpropagate() {
    //ActivationLayer_BP_Kernal<activationFunction><<<blocks, threads,0, streamProvider->get(0)>>>
    //    (getOutput(), getErrorGradients(), outErrorGradients, getOutputShape().size());
}

/*
template<class activationType>
__global__ void testKernal(float val) {
    printf("testKernal: %f\n", activationType::activate(val));
}


template<class activationType>
class ActivationLayer {
public:
    void Feed() {
        testKernal<activationType><<<1,1>>>(13);
    }
};


class Sigmoid {
public:
    __device__ static float activate(const float val){
        printf("activate return: %f\n", val+1);
        return val+1;
    }
};*/
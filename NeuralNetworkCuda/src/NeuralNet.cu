#include "NeuralNet.cuh"
#include <thread>
#include <vector>


NeuralNet::NeuralNet() {
}



void NeuralNet::setWeightsInitFunc(std::vector<val_t> (*newFunc)(size_t size)) { 
    for(auto& layer: layers) {
        layer->setWeightsInitFunc(newFunc);
    }
}

void NeuralNet::setBiosInitFunc(std::vector<val_t> (*newFunc)(size_t size)) { 
    for(auto& layer: layers) {
        layer->setBiosInitFunc(newFunc);
    }
}

void NeuralNet::RunOnce(LossFunction& lossFunc, const val_t* inputs, const val_t* expectedOutput) {
    FeedForward(inputs);
    BackPropagate(lossFunc, expectedOutput);
}

void NeuralNet::Init(const Shape3d& inputShape) {
    assert(layers.size() > 0);
    streamProvider.Init(1, 2);

    devAlloc((void**)&inputBuffer, inputShape.size()*sizeof(val_t));
    layers[0]->Init(inputShape);
    layers[0]->setInput(inputBuffer);

    for(int i = 0;i < layers.size()-1; i++) {
        layers[i+1]->Init(layers[i]->getOutputShape());
        layers[i+1]->setInput(layers[i]->getOutput());
        layers[i]->setErrorGradient(layers[i+1]->getOutErrorGradients());
        //layers[i]->setCudaStream(&cudaStream);
        layers[i]->setStreamProvider(&streamProvider);
    }
    devAlloc((void**)&errorGradientsBuffer, sizeof(val_t)*layers.back()->getOutputShape().size());
    devAlloc((void**)&dev_expectedOutput,   sizeof(val_t)*layers.back()->getOutputShape().size());
    layers.back()->setErrorGradient(errorGradientsBuffer);
    //layers.back()->setCudaStream(&cudaStream);
    layers.back()->setStreamProvider(&streamProvider);
}

void NeuralNet::getResult(val_t* resutlsBuffer) {
    cudaStreamSynchronize(streamProvider.get(0));
    cudaMemcpy(resutlsBuffer, layers.back()->getOutput(), 
        sizeof(val_t)*layers.back()->getOutputShape().size(), cudaMemcpyDeviceToHost);
}

void NeuralNet::BackPropagate(const val_t* errorGradients) {
    NopLoss nop;
    BackPropagate(nop, errorGradients);
}

void NeuralNet::BackPropagate(LossFunction& loss, const val_t* expectedOutput) {
    
    cudaMemcpy(dev_expectedOutput, expectedOutput, sizeof(val_t)*layers.back()->getOutputShape().size(), cudaMemcpyHostToDevice);
    loss.df(layers.back()->getOutput(), dev_expectedOutput, errorGradientsBuffer,   layers.back()->getOutputShape().size());
    batchError = loss.f(layers.back()->getOutput(), dev_expectedOutput,             layers.back()->getOutputShape().size(), 0);

    BackPropagate();
}

void NeuralNet::BackPropagate() {
    for(int i = layers.size()-1; i >= 0;i--) {
        layers[i]->Backpropagate();
    }
}

void NeuralNet::FeedForward(const std::vector<val_t>& vec) {
    assert(vec.size() == layers[0]->getInputShape().size());
    FeedForward(vec.data());
}

void NeuralNet::FeedForward(const val_t* input) {
    cudaDeviceSynchronize();
    cudaMemcpy(inputBuffer, input, sizeof(val_t)*layers[0]->getInputShape().size(), cudaMemcpyHostToDevice);

    FeedForward();
}

__global__ void updateWeightsKernal(val_t* weights, val_t* deltas, size_t size) {
    for(size_t index = (blockIdx.x * blockDim.x + threadIdx.x); index < size; index += gridDim.x*blockDim.x) {
        weights[index] += deltas[index];
        deltas[index] = 0;
    }
}

void NeuralNet::updateWeights(size_t batchSize) {
    for(int i = 0;i < layers.size();i++) {
        {
            val_t* weights = layers[i]->getWeights();
            val_t* deltas = layers[i]->getDeltaWeights();
            size_t weightsSize = layers[i]->getWeightsSize();
            if(weights != NULL && deltas != NULL) {
                size_t threads = DEFAULT_THREAD_SIZE;
                size_t blocks = min((int)ceilf(weightsSize/(val_t)threads), DEFAULT_BLOCK_SIZE);
                updateWeightsKernal<<<blocks, threads,0, 0>>>(weights, deltas, weightsSize);
            }
        }

        {
            val_t* bioss = layers[i]->getBios();
            val_t* biossDeltas = layers[i]->getDeltaBioss();
            size_t biosSize = layers[i]->getBiosSize();
            if(bioss != NULL && biossDeltas != NULL) {
                size_t threads = DEFAULT_THREAD_SIZE;
                size_t blocks = min((int)ceilf(biosSize/(val_t)threads), DEFAULT_BLOCK_SIZE);
                updateWeightsKernal<<<blocks, threads,0, 0>>>(bioss, biossDeltas, biosSize);
            }
        }
    }
}

void NeuralNet::FeedForward() {
    for(auto& layer: layers) {
        layer->FeedForward();
    }
}


std::shared_ptr<Layer> NeuralNet::operator[](const int index) {
    return layers[index];
}

#pragma once
#include "Layer.cuh"
#include "Utils.cuh"
#include <vector>
#include <memory>
#include "DataAdapters.cuh"
#include "Optimizer.cuh"
#include "LossFunctions.cuh"

class NeuralNet {
public:
    NeuralNet();
    template<class LayerType>
    NeuralNet& operator<<(const LayerType& layer);
    void Init(const Shape3d& inputShape);
    void FeedForward(const std::vector<val_t>& vec);
    void FeedForward(const val_t* input);
    void FeedForward();
    void BackPropagate(LossFunction& loss, const val_t* expectedOutput); // calc errorGradient using the loss func
    void BackPropagate(const val_t* errorGradients); // copy memory to device and backpropagating
    void BackPropagate(); // backpropagating using the device errorgradients memory
    void RunOnce(LossFunction& lossFunc, const val_t* inputs,
         const val_t* expectedOutput);
    val_t batchError;

    void setBiosInitFunc   (std::vector<val_t> (*newFunc)(size_t size));
    void setWeightsInitFunc(std::vector<val_t> (*newFunc)(size_t size));
    void getResult(val_t* resutlsBuffer);
    std::shared_ptr<Layer> operator[](const int index);

    template<typename OnBatchEnumerate,
             typename OnEpochEnumerate>
    void Train(optimizer& opt,LossFunction& lossFunc, DataAdapter& dataAdapter, int epocs, size_t beachSize,
         OnBatchEnumerate on_batch_enumerate, OnEpochEnumerate on_epoch_enumerate);


private:
    void updateWeights(size_t batchSize);
    size_t maxBeachSize;
    val_t* inputBuffer = nullptr;
    val_t* errorGradientsBuffer = nullptr;
    val_t* dev_expectedOutput = nullptr;
    std::vector<std::shared_ptr<Layer>> layers;
    
    //cudaStream_t cudaStream;
    CudaStreamProvider streamProvider;
};


template<class LayerType>
NeuralNet& NeuralNet::operator<<(const LayerType& layer) {
    std::shared_ptr<LayerType> ptr = std::make_shared<LayerType>(layer);
    layers.push_back(ptr);
    return *this;
}


template<typename OnBatchEnumerate,
         typename OnEpochEnumerate>
void NeuralNet::Train(optimizer& opt, LossFunction& lossFunc, DataAdapter& dataAdapter, const int epocs, const size_t beachSize,
OnBatchEnumerate on_batch_enumerate, OnEpochEnumerate on_epoch_enumerate) {
    dataAdapter.reset();
    int currentBatch = beachSize;
    for(int epoc = 0; epoc < epocs; epoc++) {
        while(dataAdapter.hasNext()) {
            DataHolder dh = dataAdapter.get();
            this->RunOnce(lossFunc, dh.inputs,dh.expectedOutputs);
            currentBatch--;
            if(currentBatch == 0) {
                updateWeights(beachSize);
                on_batch_enumerate();
                currentBatch = beachSize;
            }
        }
        on_epoch_enumerate();
        dataAdapter.reset();
    }
}
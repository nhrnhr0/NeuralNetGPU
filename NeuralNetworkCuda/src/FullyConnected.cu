#include "FullyConnected.cuh"
#include "Layer.cuh"
#include "Utils.cuh"


FullyConnected::FullyConnected(const Shape3d& outputShape, bool useBios): Layer(), useBios(useBios) {
    setOutputShape(outputShape);
}

// not a deep copy
FullyConnected::FullyConnected(const FullyConnected& other): Layer(other), useBios(other.useBios) {
    //printf("FullyConnected copy\n");
}

void FullyConnected::Init(const Shape3d& inputShape) {
    setInputShape(inputShape);
    
    devAlloc((void**)&output,            sizeof(val_t) *outputShape.size());
    devAlloc((void**)&outErrorGradients, sizeof(val_t) * inputShape.size());

    this->weightsSize = inputShape.size()*outputShape.size();

    devAlloc((void**)&weights,      sizeof(val_t)*weightsSize);
    devAlloc((void**)&deltas,       sizeof(val_t)*weightsSize);

    cudaMemset(output,            0, sizeof(val_t) * outputShape.size());
    cudaMemset(outErrorGradients, 0, sizeof(val_t) * inputShape.size());
    cudaMemset(weights,           0, sizeof(val_t) * weightsSize);
    cudaMemset(deltas,            0, sizeof(val_t) * weightsSize);


    if(useBios) {
        this->biosWeightsSize = outputShape.size();
        devAlloc((void**)&biosWeights,   sizeof(val_t)*biosWeightsSize);
        devAlloc((void**)&biosDeltas,    sizeof(val_t)*biosWeightsSize);
    }
     

    initWeights();
    if(useBios)
        initBios();
}

__global__ void FullyConnected_FF_Kernal(const val_t* input, const val_t* weights, val_t* output, size_t inputSize, val_t* bios) {
    extern __shared__ val_t sdata[];
    
    const size_t weightOffset = inputSize*blockIdx.x;

    sdata[threadIdx.x] = 0;

    // for every block (outputNeuron) has sdata.
    // sdata is filled with input*weights.
    for(int i = threadIdx.x; i < inputSize; i+= blockDim.x) {
        sdata[threadIdx.x] += 
            input[i] *
            weights[weightOffset + i];
        __syncthreads();
    }

    // sum sdata => sdata[0]:
    for(size_t s = blockDim.x/2; s>0;s>>=1) {
        if(threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        output[blockIdx.x] = sdata[0];
        if(bios!= nullptr)
            output[blockIdx.x] += 1 * bios[blockIdx.x];
    }
}


void FullyConnected::FeedForward() {
    size_t threads = DEFAULT_THREAD_SIZE;
    size_t blocks = outputShape.size();
    FullyConnected_FF_Kernal<<<blocks, threads, threads*sizeof(val_t), streamProvider->get(0)>>>
        (input, weights, output, inputShape.size(), useBios? biosWeights: nullptr);
}

__global__ void FullyConnected_BP_OUT_Kernal(const val_t* errorGradients, const val_t* weights, val_t* outErrorGradients, size_t errGradientsSize) {
    extern __shared__ val_t sdata[];
    sdata[threadIdx.x] = 0;
    for(size_t i = threadIdx.x; i < errGradientsSize; i += blockDim.x) {
        sdata[threadIdx.x] += 
        errorGradients[i] *
            weights[gridDim.x * i + blockIdx.x];
        __syncthreads();
    }

    // sum sdata => sdata[0]:
    for(size_t s = blockDim.x/2; s>0;s>>=1) {
        if(threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        outErrorGradients[blockIdx.x] = sdata[0];
    }
}

__global__ void FullyConnected_BP_Kernal(
    const val_t* input, const val_t* errorGradients, val_t* deltas, val_t* biosDeltas, size_t inputSize) {
    
    for(size_t i = threadIdx.x; i < inputSize; i+= blockDim.x) {
        size_t weightIdx = blockIdx.x*inputSize + i;
        deltas[weightIdx] += input[i] * errorGradients[blockIdx.x];
    }

    if(biosDeltas != nullptr && threadIdx.x == 0) {
        biosDeltas[blockIdx.x] += 1 * errorGradients[blockIdx.x];
    }
}

void FullyConnected::Backpropagate() {
    const size_t threads1 = DEFAULT_THREAD_SIZE;
    const size_t blocks1 = getInputShape().size();

    FullyConnected_BP_OUT_Kernal<<<blocks1, threads1, threads1*sizeof(val_t), streamProvider->get(0)>>>
        (errorGradients, weights, outErrorGradients, getOutputShape().size());
        
    const size_t threads2 = DEFAULT_THREAD_SIZE;
    const size_t blocks2 = getOutputShape().size();

    FullyConnected_BP_Kernal<<<blocks2, threads2, 0, streamProvider->get(0,0)>>>
        (input, errorGradients, deltas, biosDeltas, getInputShape().size());
    streamProvider->wait(0);
}


void FullyConnected::BackpropagateCPU() {
    for(size_t outputIdx= 0; outputIdx < outputShape.size();outputIdx++) {
        for(size_t inputIdx = 0; inputIdx < inputShape.size();inputIdx++) {
            size_t weightIdx = outputIdx*inputShape.size() + inputIdx;
            deltas[weightIdx] += input[inputIdx] * errorGradients[outputIdx];
            //printf("delt[%d] %f = inp[%d] %f * err[%d] %f\n",
            //    weightIdx, deltas[weightIdx], inputIdx, input[inputIdx], outputIdx, errorGradients[outputIdx]);
        }
        if(useBios) {
            biosDeltas[outputIdx] += 1 * errorGradients[outputIdx];
        }
    }

    for(size_t inputIdx = 0; inputIdx < inputShape.size();inputIdx++) {
        outErrorGradients[inputIdx] = GetPrevErrorGradient(inputIdx);
    }
}

val_t FullyConnected::GetPrevErrorGradient(size_t prevIdx) {
    val_t sum = 0.0;
    size_t weightsPerOutput = inputShape.size();
    for(size_t outputIdx = 0; outputIdx < outputShape.size(); outputIdx++)  {
        const size_t weightIdx = prevIdx + outputIdx * weightsPerOutput;
        sum += errorGradients[outputIdx] * weights[weightIdx];
    }
    return sum;
}
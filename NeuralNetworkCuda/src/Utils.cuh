#pragma once

#include <vector>
#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <curand.h>
#include <math.h>
#include <chrono>
#include <string>
#include <map>
#include <cstdint>


void devAlloc(void** devPtr, size_t size);
void devDelete(void* devPtr);
typedef float val_t;

val_t getRand(val_t min, val_t max);
std::vector<val_t> randomVector(size_t size);
std::vector<val_t> randomVector(size_t size, val_t min, val_t max);

uint32_t intFromBytes(unsigned char* bytes);

/*
template<typename Func>
__global__ void forIKernal(size_t size, Func func);

template<typename Func> 
void forI(size_t size, Func func);
*/
void devRandArr(val_t* arr, size_t size, val_t min = 0, val_t max = 1);

class Shape3d {
public:
    Shape3d(size_t depth, size_t height, size_t width);
    Shape3d(size_t height, size_t width);
    Shape3d(size_t width);
    Shape3d();
    Shape3d(const Shape3d& other);
    size_t size()const;

private:
    size_t depth, height, width;
};


struct Times {
    double maxTime = std::numeric_limits<double>::lowest();
    double minTime = std::numeric_limits<double>::max();
    double totalTime = 0;
    clock_t startTime;
    size_t epocs = 0;
    bool isRunning = false;
};

class Timer {
    
private:
    static std::map<std::string, struct Times> timers;

public:
    static void start(const std::string& timerName);
    static void stop(const std::string& timerName);
    static void abort(const std::string& timerName);
    static void printTimers();
};


class CudaStreamProvider {
    
public:
    CudaStreamProvider();
    void Init(size_t maxMainStreams, size_t maxSubStreams);
    cudaStream_t get(size_t streamIdx);
    cudaStream_t get(size_t streamIdx, size_t subStreamIdx);
    void wait(size_t streamIdx);

    size_t getMaxSubStreams()const;
    size_t getMaxMainStreams()const; 

private:
    std::vector<cudaStream_t> mainStreams;
    std::vector<std::vector<cudaStream_t>> subStreams;
    cudaEvent_t event;

    size_t maxMainStreams;
    size_t maxSubStreams;

};

/*

template<typename Func>
__global__ void forIKernal(size_t size, Func func) {
    for(size_t index = (blockIdx.x * blockDim.x + threadIdx.x); index < size; index += gridDim.x*blockDim.x) {
        func(index);
    }
}

template<typename T, typename Func> 
__global__ void forEKernal(T* arr, size_t size, Func func) {
    for(size_t index = (blockIdx.x * blockDim.x + threadIdx.x); index < size; index += gridDim.x*blockDim.x) {
        //printf("KernalE calls func(%p, %d)\n", arr, index);
        func(arr, index);
    }
}*/

#define DEFAULT_THREAD_SIZE 32
#define DEFAULT_BLOCK_SIZE 5

/*template<typename Func> 
void forI(size_t size, Func func) {
    size_t threads = DEFAULT_THREAD_SIZE; 
    
    size_t blocks = std::min(std::ceil(size/(float)threads), (float)DEFAULT_BLOCK_SIZE);
    printf("forIKernal: bl.:%lu Th.:%lu\n", blocks, threads);
    forIKernal<<<blocks, threads, 0>>>(size, func);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("ForI error %d\n", err);
    }else {
        printf("ForI done\n");
    }
}


template<typename T, typename Func> 
void forE(T* arr, size_t size, Func func) {
    size_t threads = 32;
    size_t blocks = std::min(std::ceil(size/(float)threads), (float)MAX_BLOCKS);
    printf("forIKernal: bl.:%lu Th.:%lu\n", blocks, threads);
    forEKernal<<<blocks, threads, 0>>>(arr, size, func);
    cudaDeviceSynchronize();
}*/
#include <iostream>
#include <stdio.h>
#include "Utils.cuh"
#include "FullyConnected.cuh"
#include "ActivationLayer.cuh"
#include "Sigmoid.cuh"
#include "NeuralNet.cuh"
#include <cuda_profiler_api.h>
#include <fstream>
#include <iostream>
#include "DataAdapters.cuh"
#include "Optimizer.cuh"
/* *

int main(int argc, char* argv[]) {
    VectorDataAdapter DataAdapter(
        {
            {0,0},
            {0,1},
            {1,0},
            {1,1}
        },
        {
            {0},{1},{1},{0}
        });
    NeuralNet net;
    net << FullyConnected(Shape3d(10), true)
        << ActivationLayer<Sigmoid>()
        << FullyConnected(Shape3d(1))
        <<  ActivationLayer<Sigmoid>();
    net.Init(Shape3d(2), 4);

    int epocs = 500;
    size_t beachSize = 4;
    //    net.Train<MSE>(gradient_descent(), DataAdapter, epocs, beachSize);
    gradient_descent opt = gradient_descent();
    MSE loss = MSE();
    net.Train(opt, loss, DataAdapter, epocs, beachSize, [&](){printf("err: %f\n", net.batchError);}, [&](){
        val_t* resultBuffer = (val_t*)malloc(sizeof(val_t)*8);
        net.FeedForward({0,0,0,1,1,0,1,1},4);
        net.getResults(resultBuffer, 4);
        printf("0 => %2.2f\n",resultBuffer[0]);
        printf("1 => %2.2f\n",resultBuffer[1]);
        printf("1 => %2.2f\n",resultBuffer[2]);
        printf("0 => %2.2f\n",resultBuffer[3]);
        delete resultBuffer;

    });
}

*/

void TrainAND() {
    size_t epocs = 100;
    size_t beachSize = 4;
    NeuralNet net;
    net << FullyConnected(Shape3d(5), true)
        << ActivationLayer<Sigmoid>()
        << FullyConnected(1)
        << ActivationLayer<Sigmoid>();
    net.Init(Shape3d(2));

    VectorDataAdapter DA = VectorDataAdapter(
        {
            {0,0},
            {0,1},
            {1,0},
            {1,1}
        },
        {
            {0},
            {1},
            {1},
            {1}
        }
    );

    gradient_descent opt = gradient_descent();
    MSE loss = MSE();

    net.Train(opt, loss, DA, epocs, beachSize, [&](){printf("err: %f\n", net.batchError);},[](){});
}
/* *
void trainMnist() {
    MnistDataAdapter mnistDA("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    size_t beachSize = 20;
    size_t epoc = 5;

    NeuralNet net;
    net << FullyConnected(Shape3d(128),true)
        << ActivationLayer<Sigmoid>()
        << FullyConnected(512, true)
        << ActivationLayer<Sigmoid>()
        << FullyConnected(10, true)
        << ActivationLayer<Sigmoid>();
    net.Init(Shape3d(28*28), beachSize);

    gradient_descent opt = gradient_descent();
    MSE loss = MSE();

    net.Train(opt, loss, mnistDA, epoc, beachSize, [&](){printf("err: %f\n", net.batchError);},[](){});
}
*/

int main(int argc, char* argv[]) {
    
    //trainMnist();
    TrainAND();

    return 0;
}

/* *
int main(int argc, char* argv[]) {
    Timer::start("main");
    cudaProfilerStart();

    std::ifstream images("train-images.idx3-ubyte", std::ios::in|std::ios::binary);
    std::ifstream lables("train-labels.idx1-ubyte", std::ios::in|std::ios::binary);
    unsigned char mem[28*28];
    std::vector<val_t> inputs;
    
    int dataOffset = 16;
    images.read(reinterpret_cast<char*>(mem), dataOffset);
    int magicNum = intFromBytes(mem);
    int imagesNum   = intFromBytes(mem+4);
    int rowSize  = intFromBytes(mem+8);
    int colSize  = intFromBytes(mem+12);
    lables.seekg(8, lables.beg);
    size_t beachSize = 20;

    NeuralNet net;
    net << FullyConnected(Shape3d(128),true)
        << ActivationLayer<Sigmoid>()
        << FullyConnected(512, true)
        << ActivationLayer<Sigmoid>()
        << FullyConnected(10, true)
        << ActivationLayer<Sigmoid>();
    net.Init(Shape3d(rowSize*colSize), beachSize);
    

    char lable;
    //imagesNum = 10000;
    std::vector<val_t> expectedOutput(10 * beachSize);
    inputs.resize(28*28*beachSize);
    expectedOutput.assign(expectedOutput.size(), 0);
    for(int i = 0;i < imagesNum; i+=beachSize) {
        for(int k = 0;k < beachSize; k++) {
            images.read(reinterpret_cast<char*>(mem), rowSize*colSize);
            for(int j = 0;j < rowSize*colSize;j++) {
                inputs.at(k * rowSize*colSize + j) = static_cast<val_t>(mem[j]) / 255.0;
            }

        
            lables.read(reinterpret_cast<char*>(&lable), 1);
            int index = static_cast<int>(lable);
            expectedOutput[k*10 + index] = 1;
        }
        net.RunBeach(inputs, expectedOutput, beachSize);
        //net.FeedForward(inputs, beachSize);
        //net.BackPropagate(expectedOutput, beachSize);
        printf("%3.2f\n", ((float)i/(float)imagesNum)*100);
    }

    Timer::stop("main");
    Timer::printTimers();
    cudaProfilerStop();
    return 0;
}
*/
# NeuralNetGPU

## Description
NeuralNetGPU is a librery for creating and training neural networks in c++
 with the efficency of the GPU using the CUDA (parallel computing platform)
 
 ## Usage
 this library lets you easly create neural networks and train them
 here is an example of how to train a network as an [AND gate](https://en.wikipedia.org/wiki/AND_gate):
 
 ```c++
 void TrainAND() {
    size_t epocs = 10;
    size_t beachSize = 4;
    NeuralNet net;
    net << FullyConnected(Shape3d(5), true)
        << ActivationLayer<Sigmoid>()
        << FullyConnected(1)
        << ActivationLayer<Sigmoid>();
    net.Init(Shape3d(2));

    VectorDataAdapter DA = VectorDataAdapter(
        { {0,0},{0,1},{1,0},{1,1} },
        { {0},{1},{1},{1} }
    );

    gradient_descent opt = gradient_descent();
    MSE loss = MSE();

    net.Train(opt, loss, DA, epocs, beachSize, [&](){printf("err: %f\n", net.batchError);},[](){});
}
 ```
 Here net will contain neural network with 2 input neurons, 5 hidden neurons and 1 output neuron,
 between the layers there is a sigmoid activation layer
 
 ## Layers
 ### FullyConnected
 all the neurons from the input shape will be connected to all the neurons in the output shape ([Cartesian product](https://he.wikipedia.org/wiki/%D7%9E%D7%9B%D7%A4%D7%9C%D7%94_%D7%A7%D7%A8%D7%98%D7%96%D7%99%D7%AA))
 ## Activation Layers
 ### Sigmoid
 [Sigmoid](https://miro.medium.com/max/1280/1*sOtpVYq2Msjxz51XMn1QSA.png)
 ## Loss Functions
 ### MSE
 Mean Square Error (MSE) is the most commonly used regression loss function.
 MSE is the sum of squared distances between our target variable and predicted values.
 
 ## Optimizers
 ### gradient descent
 Gradient descent is an optimization algorithm used to minimize some function by iteratively 
 moving in the direction of steepest descent as defined by the negative of the gradient. 
 
 

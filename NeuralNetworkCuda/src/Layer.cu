#include "Layer.cuh"


Layer::Layer(): 
    input(nullptr),
    output(nullptr),
    errorGradients(nullptr),
    outErrorGradients(nullptr),
    weights(nullptr),
    deltas(nullptr),
    biosWeights(nullptr),
    biosDeltas(nullptr),
    weightsSize(0),
    biosWeightsSize(0)
{}

// not a deep copy
Layer::Layer(const Layer& other) {
    inputShape = other.inputShape;
    outputShape = other.outputShape;
    //printf("Layer copy\n");
}

void Layer::setInputShape(const Shape3d& shape){
    this->inputShape = Shape3d(shape);
}

void Layer::setOutputShape(const Shape3d& shape) {
    this->outputShape = Shape3d(shape);
}

void Layer::setInput(const val_t* input) {
    this->input = input;
}

void Layer::setErrorGradient(const val_t* errorGradient) {
    this->errorGradients = errorGradient;
}

const val_t* Layer::getOutput()const {
    return output;
}

const val_t* Layer::getOutErrorGradients()const {
    return outErrorGradients;
}

const val_t* Layer::getInput()const {
    return input;
}

const val_t* Layer::getErrorGradients()const {
    return errorGradients;
}

const Shape3d& Layer::getInputShape()const {
    return inputShape;
}

bool Layer::isUseBios()const {
    return !(biosWeights == nullptr);
}

const Shape3d& Layer::getOutputShape()const {
    return outputShape;
}

Layer::~Layer() {
    if(output)
        devDelete(output);
    if(outErrorGradients)
        devDelete(outErrorGradients);
    if(weights)
        devDelete(weights);
    if(deltas)
        devDelete(deltas);
    if(biosWeights)
        devDelete(biosWeights);
    if(biosDeltas)
        devDelete(biosDeltas);
}

void Layer::setStreamProvider(CudaStreamProvider* streamProvider) {
    this->streamProvider = streamProvider;
}

void Layer::setWeightsInitFunc(std::vector<val_t> (*newFunc)(size_t size)) {
    weightsInitFunc = newFunc;
}

void Layer::setBiosInitFunc(std::vector<val_t> (*newFunc)(size_t size)) {
    biosInitFunc = newFunc;
}

void Layer::initWeights() {
    cudaMemcpy(weights, weightsInitFunc(weightsSize).data(), 
        sizeof(val_t)*weightsSize, cudaMemcpyHostToDevice);
}

void Layer::initBios() {
    cudaMemcpy(biosWeights, biosInitFunc(biosWeightsSize).data(), 
        sizeof(val_t)*biosWeightsSize, cudaMemcpyHostToDevice);
}



val_t* Layer::getWeights() {
    return weights;
}
val_t* Layer::getBios() {
    return biosWeights;
}
val_t* Layer::getDeltaWeights() {
    return deltas;
}
val_t* Layer::getDeltaBioss() {
    return biosDeltas;
}
size_t Layer::getWeightsSize() const{
    return weightsSize;
}
size_t Layer::getBiosSize() const {
    return biosWeightsSize;
}
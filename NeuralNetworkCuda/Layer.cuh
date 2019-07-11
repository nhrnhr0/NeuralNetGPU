#pragma once
#include <vector>

#include "Utils.cuh"
class Layer {
public:
	friend class NeuralNet;
	Layer();
	Layer(const Layer& other);
	~Layer();
	void setInput(const val_t* input);
	void setErrorGradient(const val_t* errorGradient);

	const val_t* getOutput()const;
	const val_t* getOutErrorGradients()const;

	const val_t* getInput()const;
	const val_t* getErrorGradients()const;

	const Shape3d& getOutputShape()const;
	const Shape3d& getInputShape()const;

	val_t* getWeights();
	val_t* getBios();
	val_t* getDeltaWeights();
	val_t* getDeltaBioss();
	size_t getWeightsSize()const;
	size_t getBiosSize()const;

	bool isUseBios()const;


	// Init needs to alocate output, outErrorGradients. also weights and deltas if necessary
	// Init need to set inputShape and outputShape
	virtual void Init(const Shape3d& inputShape) = 0;
	virtual void FeedForward() = 0;
	virtual void Backpropagate() = 0;

	void setWeightsInitFunc(std::vector<val_t>(*newFunc)(size_t size));
	void setBiosInitFunc(std::vector<val_t>(*newFunc)(size_t size));

	void setStreamProvider(CudaStreamProvider* streamProvider);

protected:
	CudaStreamProvider* streamProvider;
	const val_t* input;
	val_t* output = nullptr;
	const val_t* errorGradients;
	val_t* outErrorGradients = nullptr;

	val_t* weights = nullptr;
	val_t* deltas = nullptr;

	val_t* biosWeights = nullptr;
	val_t* biosDeltas = nullptr;

	Shape3d inputShape, outputShape;

	//cudaStream_t* cudaStream;



	void setInputShape(const Shape3d& shape);
	void setOutputShape(const Shape3d& shape);

	size_t weightsSize = 0, biosWeightsSize = 0;
	void initWeights();
	void initBios();
	std::vector<val_t>(*weightsInitFunc)(size_t size) = randomVector;
	std::vector<val_t>(*biosInitFunc)(size_t size) = randomVector;
};

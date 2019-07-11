#pragma once

#include "Layer.cuh"
class FullyConnected : public Layer {

public:
	FullyConnected(const Shape3d& outputShape, bool useBios = true);
	FullyConnected(const FullyConnected& other);
	virtual void Init(const Shape3d& inputShape)override;
	virtual void FeedForward()override;
	virtual void Backpropagate()override;
	void BackpropagateCPU();

private:
	bool useBios;
	val_t GetPrevErrorGradient(size_t prevIdx);
};

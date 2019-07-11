#include "Optimizer.cuh"
__global__ void gradient_descent_kernal(const val_t* deltas, val_t* weigts, size_t weightsSize, val_t alpha, val_t lambda) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < weightsSize;
		i += blockDim.x * gridDim.x) {
		weigts[i] = weigts[i] - alpha * (deltas[i] + lambda * weigts[i]);
	}
}

void gradient_descent::update(const val_t* deltas, val_t* weights, size_t weightsSize, cudaStream_t stream) {
	size_t threads = DEFAULT_THREAD_SIZE;
	size_t blocks = std::ceil(weightsSize / (float)threads);
	gradient_descent_kernal << <blocks, threads, 0, stream >> > (deltas, weights, weightsSize, alpha, lambda);
}
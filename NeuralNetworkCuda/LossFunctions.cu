#include "LossFunctions.cuh"



__global__ void MSE_F_Kernal(const val_t* output, const val_t* expectedOutput, size_t size, val_t* res) {
	extern __shared__ val_t sdata[];
	sdata[threadIdx.x] = 0;
	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		sdata[threadIdx.x] += (output[i] - expectedOutput[i]) * (output[i] - expectedOutput[i]);
	}
	// sum sdata => sdata[0]:
	for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
	*res = sdata[0];
}

MSE::MSE() {
	devAlloc((void**)& d_val, sizeof(val_t));
}
val_t MSE::f(const val_t* output, const val_t* expectedOutput,
	const size_t size, cudaStream_t stream) {

	MSE_F_Kernal << <1, DEFAULT_THREAD_SIZE, sizeof(val_t)* DEFAULT_THREAD_SIZE, stream >> > (output, expectedOutput, size, d_val);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("MSE_F_Kernal err %d\n", err);
	}
	cudaStreamSynchronize(stream);
	val_t h_val;
	cudaMemcpy(&h_val, d_val, sizeof(val_t), cudaMemcpyDeviceToHost);
	return h_val / (val_t)size;
}


__global__ void MSE_DF_Kernal(const val_t* output, const val_t* expectedOutput, val_t* errorGradientsBuffer, const size_t size, const val_t factor) {
	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		errorGradientsBuffer[i] = factor * (output[i] - expectedOutput[i]);
	}
}

void MSE::df(const val_t* output, const val_t* expectedOutput,
	val_t* errorGradientsBuffer, const size_t size, cudaStream_t stream) {
	val_t factor = (val_t)2 / (val_t)size;
	MSE_DF_Kernal << <1, DEFAULT_THREAD_SIZE, 0, stream >> > (output, expectedOutput, errorGradientsBuffer, size, factor);

	/* *for(int i = 0;i < size;i++) {
		printf("errGradients[%d] = %2.2f\n", i, errorGradientsBuffer[i]);
	}*/
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("MSE DF kernal error %d\n", err);
	}
}



val_t NopLoss::f(const val_t* output, const val_t* expectedOutput, const size_t size, cudaStream_t stream) {
	return 0;
}

void NopLoss::df(const val_t* output, const val_t* expectedOutput,
	val_t* errorGradientsBuffer, const size_t size, cudaStream_t stream) {
	cudaMemcpy(errorGradientsBuffer, expectedOutput, sizeof(val_t) * size, cudaMemcpyDeviceToDevice);
}

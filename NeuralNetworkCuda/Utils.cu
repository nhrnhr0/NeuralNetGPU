#include "Utils.cuh"
#include <curand.h>
#include <algorithm>
#include <cuda_profiler_api.h>

size_t Shape3d::size()const { return depth * height * width; }
Shape3d::Shape3d() : Shape3d(1) {}
Shape3d::Shape3d(size_t width) : Shape3d(1, width) {}
Shape3d::Shape3d(size_t height, size_t width) : Shape3d(1, height, width) {}
Shape3d::Shape3d(size_t depth, size_t height, size_t width) : depth(depth), height(height), width(width) {}

Shape3d::Shape3d(const Shape3d& other) : Shape3d(other.depth, other.height, other.width) { }



val_t getRand(val_t min, val_t max) {
	return ((rand() / (val_t)RAND_MAX) * (max - min)) + min;
}

std::vector<val_t> randomVector(size_t size, val_t min, val_t max) {
	std::vector<val_t> ret(size);
	for (int i = 0; i < size; i++) {
		ret[i] = getRand(min, max);
	}
	return ret;
}

std::vector<val_t> randomVector(size_t size) {
	return randomVector(size, -1, 1);
}

uint32_t intFromBytes(unsigned char* bytes) {
	return (uint32_t)bytes[0] << 24 |
		(uint32_t)bytes[1] << 16 |
		(uint32_t)bytes[2] << 8 |
		(uint32_t)bytes[3];
}


void devAlloc(void** devPtr, size_t size) {
	//cudaError_t err = cudaMalloc(devPtr, size);
	cudaError_t err = cudaMallocManaged(devPtr, size);
	if (err != cudaSuccess) {
		printf("error alocating memory of size: %ld\n", size);
	}
}

void devDelete(void* devPtr) {
	cudaError err = cudaFree(devPtr);
	if (err != cudaSuccess) {
		printf("error %d to free memory: %p\n", cudaGetLastError(), devPtr);
	}
}

struct cuRandData {
	curandGenerator_t gen;
	bool isInit = false;
};
struct cuRandData cuRand;

__global__ void randomInRange_Kernal(val_t* arr, val_t min, val_t max, size_t size) {
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
		auto range = (max - min);
		range *= range > 0 ? 1 : -1;

		arr[idx] =
			range * arr[idx] +
			min;
	}
}


/**
void devRandArr(val_t* arr, size_t size, val_t min, val_t max) {
	if (arr == nullptr || size == 0)
		return;
	if (cuRand.isInit == false) {
		curandCreateGenerator(&cuRand.gen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(cuRand.gen, 1234ULL);
		cuRand.isInit = true;
	}

	if (typeid(val_t) == typeid(float)) {
		curandGenerateUniform(cuRand.gen, (float*)arr, size);
	}
	else {
		curandGenerateUniformDouble(cuRand.gen, (double*)arr, size);
	}

	int threads = DEFAULT_THREAD_SIZE;
	int blocks = std::min((int)ceilf(size / (float)threads), DEFAULT_THREAD_SIZE);
	randomInRange_Kernal << <blocks, threads >> > (arr, min, max, size);
	cudaDeviceSynchronize();
}
*/



std::map<std::string, struct Times> Timer::timers;

void Timer::abort(const std::string& timerName) {
	auto& timer = timers[timerName];
	if (timer.isRunning == false) {
		printf("%s timer can't be aborted without been started\n", timerName.data());
		exit(1);
	}
	timer.isRunning = false;
	//timer.currentTimer = clock();
}

void Timer::start(const std::string& timerName) {
	auto& timer = timers[timerName];
	if (timer.isRunning) {
		printf("%s timer was already started\n", timerName.data());
		exit(1);
	}
	else {
		timer.startTime = clock();
		timer.isRunning = true;
	}
}


void Timer::stop(const std::string& timerName) {
	auto& timer = timers[timerName];
	if (timer.isRunning == false) {
		printf("%s timer can't stop if it wasn't started\n", timerName.data());
		exit(1);
	}
	double duration = (clock() - timer.startTime) / (double)CLOCKS_PER_SEC;
	if (duration > timer.maxTime) {
		timer.maxTime = duration;
	}
	if (duration < timer.minTime) {
		timer.minTime = duration;
	}

	timer.totalTime += duration;
	timer.epocs++;
	timer.isRunning = false;
	//timer.currentTimer = time_since_epoc();
}

void Timer::printTimers() {
	for (auto timePair : timers) {
		printf("%s => arv: %lf min: %lf max: %lf\n",
			timePair.first.data(),
			timePair.second.totalTime / timePair.second.epocs,
			timePair.second.minTime,
			timePair.second.maxTime);

	}
}



cudaStream_t CudaStreamProvider::get(size_t streamIdx) {
	assert(streamIdx < maxMainStreams);

	return mainStreams.at(streamIdx);
}
cudaStream_t CudaStreamProvider::get(size_t streamIdx, size_t subStreamIdx) {
	assert(streamIdx < maxMainStreams);
	assert(subStreamIdx < maxSubStreams);

	return subStreams[streamIdx][subStreamIdx];
}

void CudaStreamProvider::wait(size_t streamIdx) {
	for (int i = 0; i < maxSubStreams; i++) {
		cudaEventRecord(event, subStreams[streamIdx][i]);
		cudaStreamWaitEvent(mainStreams[streamIdx], event, 0);
	}
}

CudaStreamProvider::CudaStreamProvider() {}

void CudaStreamProvider::Init(size_t maxMainStreams, size_t maxSubStreams) {
	this->maxMainStreams = maxMainStreams;
	this->maxSubStreams = maxSubStreams;
	mainStreams.resize(maxMainStreams);
	subStreams.resize(maxMainStreams);
	for (int i = 0; i < maxMainStreams; i++) {
		cudaStreamCreate(&mainStreams[i]);
		subStreams[i].resize(maxSubStreams);
		for (int j = 0; j < subStreams[i].size(); j++) {
			cudaStreamCreate(&subStreams[i][j]);
		}
	}
	cudaEventCreate(&event, cudaEventDisableTiming);
}

size_t CudaStreamProvider::getMaxSubStreams()const {
	return maxSubStreams;
}
size_t CudaStreamProvider::getMaxMainStreams()const {
	return maxMainStreams;
}
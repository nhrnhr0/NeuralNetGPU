#pragma once
#include <vector>
#include "Utils.cuh"
#include <fstream>

struct DataHolder {
	val_t* inputs;
	val_t* expectedOutputs;
};

class DataAdapter {
public:
	DataAdapter() = default;
	virtual void reset() = 0;
	virtual bool hasNext() = 0;
	virtual DataHolder get() = 0;
};

class VectorDataAdapter : public DataAdapter {
public:
	VectorDataAdapter(const std::vector<std::vector<val_t>>& inputs, const std::vector<std::vector<val_t>>& expectedOutputs);
	void reset()override;
	bool hasNext()override;
	DataHolder get()override;
private:
	std::vector<val_t> _inputs;
	std::vector<val_t> _outputs;
	size_t singleInputSize, singleOutputSize;
	size_t currentEntry = 0;
};
/* *
class MnistDataAdapter : public DataAdapter {
public:
	MnistDataAdapter(const std::string imagesFileName, const std::string lablesFileName);
	void reset()override;
	bool hasNext()override;
	DataHolder get(size_t maxSize)override;

private:
	std::ifstream imagesFS;
	std::ifstream lablesFS;
	const size_t maxImages = 50;
	int imagesNum;
	int currImage;
	std::vector<val_t> imagesBuffer;
	std::vector<val_t> lablesBuffer;
	unsigned char mem[28*28];
};
*/
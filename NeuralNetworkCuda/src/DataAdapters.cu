#include "DataAdapters.cuh"


VectorDataAdapter::VectorDataAdapter(const std::vector<std::vector<val_t>>& inputs, const std::vector<std::vector<val_t>>& outputs) {
    assert(inputs.size() == outputs.size() && inputs.size() > 0);
    singleInputSize = inputs[0].size();
    singleOutputSize = outputs[0].size();
    _inputs.reserve(inputs.size()*inputs[0].size());
    _outputs.reserve(outputs.size()*outputs[0].size());
    for(int i = 0;i < inputs.size();i++) {
        _inputs.insert(_inputs.end(), inputs[i].begin(), inputs[i].end());
        _outputs.insert(_outputs.end(), outputs[i].begin(), outputs[i].end());
    }
}

void VectorDataAdapter::reset() {
    currentEntry = 0;
}

bool VectorDataAdapter::hasNext() {
    return currentEntry < (_inputs.size()/singleInputSize);
}

DataHolder VectorDataAdapter::get() {
    DataHolder dh;
    dh.inputs = _inputs.data() + currentEntry * singleInputSize;
    dh.expectedOutputs = _outputs.data() + currentEntry * singleOutputSize;
    currentEntry++;
    return dh;
}

/* *
MnistDataAdapter::MnistDataAdapter(const std::string imagesFileName, const std::string lablesFileName) : 
     imagesFS(imagesFileName.c_str(), std::ios::in|std::ios::binary),
     lablesFS(lablesFileName.c_str(), std::ios::in|std::ios::binary) {

    //imagesBuffer.resize(28*28*maxImages);
    //lablesBuffer.resize(10 *  maxImages);
    int dataOffset = 16;

    imagesFS.read(reinterpret_cast<char*>(mem), dataOffset);
    int magicNum = intFromBytes(mem);
    imagesNum   = intFromBytes(mem+4);
    currImage = 0;
    int rowSize  = intFromBytes(mem+8);
    int colSize  = intFromBytes(mem+12);
    lablesFS.seekg(8, lablesFS.beg);
}


bool MnistDataAdapter::hasNext() {
    return currImage < imagesNum;
}

void MnistDataAdapter::reset() {
    lablesFS.seekg(8, lablesFS.beg);
    imagesFS.seekg(16, imagesFS.beg);
    currImage = 0;
}

DataHolder MnistDataAdapter::get(size_t maxSize) {
    char lable;
    imagesBuffer.clear();
    imagesBuffer.reserve(maxSize*28*28);
    lablesBuffer.resize(maxSize*10);
    lablesBuffer.assign(lablesBuffer.size(), 0); // set all to 0
    int currBeach = 0;
    for(int i = 0;i < maxSize; i++) {
        imagesFS.read(reinterpret_cast<char*>(mem), 28*28);
        for(int j = 0; j < 28*28; j++) {
            imagesBuffer.push_back(static_cast<val_t>(mem[j]) / 255.0f);
        }
        lablesFS.read(reinterpret_cast<char*>(&lable), 1);
        int index = static_cast<int>(lable);
        lablesBuffer[i*10 + index] = 1;
        currImage++;
        currBeach = i+1;
        if(hasNext() == false) {
            break;
        }
    }

    for(int i = 0;i < imagesBuffer.size(); i++) {
        imagesBuffer[i] = imagesBuffer[i] / 255.0f;
    }
    
    DataHolder ret;
    ret.entriesSize = currBeach;
    ret.inputs = imagesBuffer.data();
    ret.expectedOutputs = lablesBuffer.data();
    return ret;
}
*/
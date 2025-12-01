#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include "../lib/SharedMemory.hpp"

class FeatureExtractor
{
public:
    void sendToDBLogger(const std::vector<SIFTFeaturesIpc> &features);
private:
    bool getFeaturesStatus();
};

#endif // FEATURE_EXTRACTOR_HPP
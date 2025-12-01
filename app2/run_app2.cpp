#include "FeatureExtractor.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    FeatureExtractor siftPublisher;
    while (true)
    {
        if(!siftPublisher.getFeaturesStatus())
            sendToDBLogger();
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}

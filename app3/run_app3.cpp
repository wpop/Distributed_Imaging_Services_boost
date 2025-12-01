#include "DataLogger.hpp"
#include <thread>
#include <chrono>

int main()
{
    DataLogger logger;
    while(true) 
    {
        std::vector<SIFTFeaturesIpc> features = logger.get();
        if (!features.empty())
        {
            logger.updateTable(features);
            SharedMemoryCleanup(); // Final cleanup of all resources
        } 
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));   
        }
    }
    SharedMemoryCleanup(); // Final cleanup of all resources
    return 0;
}

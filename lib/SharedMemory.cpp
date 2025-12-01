#include "SharedMemory.hpp"
#include <iostream>

void SharedMemoryCleanup() {
    try {
        named_mutex::remove("SharedMutex");
        named_condition::remove("CondImages");
        named_condition::remove("CondFeatures");
        shared_memory_object::remove("ImageSharedMemory");
        shared_memory_object::remove("FeatureSharedMemory");
    } catch (const interprocess_exception& e) {
        std::cerr << "Cleanup error: " << e.what() << std::endl;
    }
}

#ifndef SHAREDMEMORY_H
#define SHAREDMEMORY_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace boost::interprocess;

// Define an STL compatible allocator that allocates from the managed_shared_memory
template <class T>
using ShmemAllocator = allocator<T, managed_shared_memory::segment_manager>;

// Define Boost.Interprocess compatible containers
template <class T>
using IpcVector = boost::interprocess::vector<T, ShmemAllocator<T>>;

// Custom data structure to store cv::Mat and SIFT features in shared memory
// cv::Mat data must be stored as a vector of chars with metadata
struct IpcMat {
    int rows = 0;
    int cols = 0;
    int type = 0;
    IpcVector<uchar> data;

    // Helper to convert IpcMat to cv::Mat
    cv::Mat toCvMat() const {
        if (rows > 0 && cols > 0 && type >= 0) {
            cv::Mat mat(rows, cols, type, (void*)data.data());
            return mat.clone(); // Clone so the new Mat owns the data if needed later outside SHM
        }
        return cv::Mat();
    }

    // Helper to convert cv::Mat to IpcMat (requires allocator)
    static IpcMat fromCvMat(const cv::Mat& mat, const ShmemAllocator<uchar>& alloc) {
        IpcMat ipcMat;
        ipcMat.rows = mat.rows;
        ipcMat.cols = mat.cols;
        ipcMat.type = mat.type();
        size_t dataSize = mat.total() * mat.elemSize();
        ipcMat.data = IpcVector<uchar>(alloc);
        ipcMat.data.resize(dataSize);
        std::memcpy(ipcMat.data.data(), mat.data, dataSize);
        return ipcMat;
    }
};

struct SIFTFeaturesIpc {
    IpcMat image;
    IpcVector<cv::KeyPoint> keypoints; // NOTE: cv::KeyPoint needs simple layout or serialization for SHM
    IpcMat descriptors;

    // Default constructor needed for placement new
    SIFTFeaturesIpc() = default;
};

// Main shared memory structure
struct SharedData {
    // Shared memory vectors must use the custom allocator
    IpcVector<IpcMat> images;
    IpcVector<SIFTFeaturesIpc> features;
    
    // Synchronization
    named_mutex mutex;
    named_condition condition_images_ready;
    named_condition condition_features_ready;
    bool imagesReady = false;
    bool featuresReady = false;

    // Constructor for placement new in shared memory
    SharedData(const ShmemAllocator<IpcMat>& alloc_images, 
               const ShmemAllocator<SIFTFeaturesIpc>& alloc_features)
        : images(alloc_images), features(alloc_features),
          mutex(open_or_create, "SharedMutex"),
          condition_images_ready(open_or_create, "CondImages"),
          condition_features_ready(open_or_create, "CondFeatures") {}
};

#endif // SHAREDMEMORY_H

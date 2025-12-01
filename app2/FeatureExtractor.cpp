#include "FeatureExtractor.hpp"
#include <iostream>

void FeatureExtractor::sendToDBLogger() {
    // Create the second shared memory segment for features
    SharedMemoryCleanup(); // Remove previous feature SHM resources for a clean run
    managed_shared_memory segment(create_only, "FeatureSharedMemory", 100 * 1024 * 1024); // 100 MB

    ShmemAllocator<IpcMat> alloc_ipc_mat(segment.get_segment_manager());
    ShmemAllocator<SIFTFeaturesIpc> alloc_ipc_features(segment.get_segment_manager());
    
    // Construct SharedData in the *features* segment (using relevant allocators)
    SharedData* data = segment.construct<SharedData>("SharedFeaturesData")(alloc_ipc_mat, alloc_ipc_features);

    {
        scoped_lock<named_mutex> lock(data->mutex);
        while (!data->imagesReady) {
            data->condition_images_ready.wait(lock);
        }
        data->features.clear();

        // the extraction happens here, using the 'segment' allocator directly
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

        // Example with actual extraction and allocation in SHM:
        managed_shared_memory segment_images(open_only, "ImageSharedMemory"); // Open the images segment to get image data
        SharedData* data_images = segment_images.find<SharedData>("SharedData").first;

        // write vector of SIFT features to Shared Memory
        for(const auto& ipcMat : data_images->images) {
            cv::Mat mat = ipcMat.toCvMat();
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            sift->detectAndCompute(mat, cv::noArray(), keypoints, descriptors);

            ShmemAllocator<uchar> alloc_uchar(segment.get_segment_manager());
            SIFTFeaturesIpc ipcFeatures;
            ipcFeatures.image = IpcMat::fromCvMat(mat, alloc_uchar);
            ipcFeatures.descriptors = IpcMat::fromCvMat(descriptors, alloc_uchar);
            
            // Allocate keypoints in SHM
            ShmemAllocator<cv::KeyPoint> alloc_keypoint(segment.get_segment_manager());
            ipcFeatures.keypoints = IpcVector<cv::KeyPoint>(alloc_keypoint);
            ipcFeatures.keypoints.resize(keypoints.size());
            std::copy(keypoints.begin(), keypoints.end(), ipcFeatures.keypoints.begin());
            
            data->features.push_back(ipcFeatures);
        }
        data->imagesReady = false; // Mark as read
        data->featuresReady = true;
        std::cout << "Modifier: Sent " << data->features.size() << " feature sets. Notifying DataLogger." << std::endl;
        data->condition_features_ready.notify_one();
    }
}

bool FeatureExtractor::getFeaturesStatus() {
    managed_shared_memory segment(open_only, "FeatureSharedMemory");
    SharedData* data = segment.find<SharedData>("SharedFeaturesData").first;
    bool status = false;
    {
        scoped_lock<named_mutex> lock(data->mutex);
        status = data->featuresReady;
    }
    return status;
}
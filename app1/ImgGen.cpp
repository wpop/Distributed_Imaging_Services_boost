#include "Sender.hpp"
#include <iostream>

void ImgGen::sendBatchImgs(const std::vector<cv::Mat>& mats) {
    SharedMemoryCleanup(); // Ensure clean state at the start of the sender
    managed_shared_memory segment(create_only, "ImageSharedMemory", 100 * 1024 * 1024); // 100MB

    ShmemAllocator<IpcMat> alloc_ipc_mat(segment.get_segment_manager());
    ShmemAllocator<SIFTFeaturesIpc> alloc_ipc_features(segment.get_segment_manager()); // Also needed for the SharedData ctor

    SharedData* data = segment.construct<SharedData>("SharedData")(alloc_ipc_mat, alloc_ipc_features);
    
    {
        scoped_lock<named_mutex> lock(data->mutex);
        data->images.clear();
        for (const auto& mat : mats) {
            // Need a specific allocator for the internal uchar vector within IpcMat
            ShmemAllocator<uchar> alloc_uchar(segment.get_segment_manager());
            data->images.push_back(IpcMat::fromCvMat(mat, alloc_uchar));
        }
        data->imagesReady = true;
        std::cout << "Sender: Sent " << data->images.size() << " images. Notifying modifier." << std::endl;
        data->condition_images_ready.notify_one();
    }
    // Shared memory remains until explicit removal
}

void ImgGen::readImgInf(const std::string &folderPath, const int batchSize)
{
    std::vector<cv::String> filenames;
    cv::glob(folderPath + "*.jpg", filenames, false);
    cv::glob(folderPath + "*.png", filenames, false);
    cv::glob(folderPath + "*.jpeg", filenames, false);
    cv::glob(folderPath + "*.bmp", filenames, false);
    if (filenames.empty())
    {
        std::cerr << "Error: No images found in the specified folder: " << folderPath << std::endl;
        return;
    }
    int currentIndex = 0;
    std::vector<cv::Mat> images;
    while (true)
    {
        std::cout << "Loading batch starting from index " << currentIndex << std::endl;
        for (int i = 0; i < batchSize; ++i)
        {
            int index = (currentIndex + i) % filenames.size(); // Wrap around if needed
            cv::Mat image = cv::imread(filenames[index]);

            if (image.empty())
            {
                std::cerr << "Warning: Could not read image: " << filenames[index] << std::endl;
                continue;
            }

            images.push_back(image);
            std::cout << "Read image: " << filenames[index] << std::endl;
        }
        currentIndex = (currentIndex + batchSize) % filenames.size(); // Update index for next batch
		sendBatchImgs(images);
		images.clear();
    }
}
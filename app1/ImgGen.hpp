#ifndef IMG_GEN_H
#define IMG_GEN_H

#include "../lib/SharedMemory.hpp"

class ImgGen
{
public:
    void readImgInf(const std::string &folderPath, const int batchSize);

private:
    void sendBatchImgs(const std::vector<cv::Mat> &mats);
};

#endif // IMG_GEN_H

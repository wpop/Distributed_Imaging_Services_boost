#include "ImgGen.hpp"

int main() 
{
    int batchSize = 3;
    ImgGen publisher;
	publisher.readImgInf("../images", batchSize);
    return 0;
}

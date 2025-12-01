cmake_minimum_required(VERSION 3.10) # Specify the minimum required CMake version
project(ImageGenerator CXX)          # Define your project name and language

find_package(OpenCV REQUIRED)
# Find Boost, specifically requesting the interprocess component
find_package(Boost REQUIRED COMPONENTS interprocess)

# Include OpenCV's header directories
iinclude_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(/usr/local/include/opencv4)
include_directories(/usr/include/opencv4)
include(/usr/include/boost/interprocess)
# include(/usr/include/boost-182/boost/interprocess)

# Add your executable target
add_executable(publish app1.cpp ImgGen.cpp)

# Link the dependencies to the main1 executable
target_link_libraries(publish PRIVATE
    ShMemLib # Link to the library created in the lib/CMakeLists.txt
    OpenCV::opencv
    Boost::interprocess
)

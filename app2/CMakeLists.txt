cmake_minimum_required(VERSION 3.10) # Specify the minimum required CMake version
project(FeatureExtractor CXX)        # Define your project name and languagec

# Find the OpenCV package
find_package(OpenCV REQUIRED)
find_package(Boost::interprocess REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(/usr/local/include/opencv4)
include_directories(/usr/include/opencv4)
include(/usr/include/boost/interprocess)

# Add your executable target
add_executable(feature app2.cpp FeatureExtractor.cpp)

# Link the dependencies to the main2 executable
target_link_libraries(feature PRIVATE
    ShMemLib
    OpenCV::opencv
    Boost::interprocess
)

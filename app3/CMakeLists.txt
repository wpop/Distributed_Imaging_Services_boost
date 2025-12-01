cmake_minimum_required(VERSION 3.10) # Specify the minimum required CMake version
project(DataLogger CXX)

find_package(OpenCV REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Boost::interprocess REQUIRED)


# Include OpenCV's header directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${SQLite3_INCLUDE_DIRS})


include_directories(/usr/local/include/opencv4)
include_directories(/usr/include/opencv4)
include(/usr/local/lib/cmake/SQLite3/SQLite3Config.cmake)
include(/usr/include/SQLite3/)
include(/usr/include/boost/interprocess)

add_executable(loger app3.cpp ImgDBManager.cpp)

# Link the dependencies to the main2 executable
target_link_libraries(loger PRIVATE
    ShMemLib # Link to the library created in the lib/CMakeLists.txt
    OpenCV::opencv
	SQLite3::SQLite3
    Boost::interprocess
)

# Link your executable with OpenCV's libraries
target_link_libraries(loger ${OpenCV_LIBS})
target_link_libraries(loger ${SQLite3_LIBRARIES})
target_link_libraries(loger ${Boost_LIBRARIES})

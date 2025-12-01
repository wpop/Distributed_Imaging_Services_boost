cmake_minimum_required(VERSION 3.10)
project(DistributedImagingServices LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Boost COMPONENTS interprocess REQUIRED)
target_link_libraries(my_app PRIVATE Boost::system Boost::filesystem)

add_subdirectory(lib)
add_subdirectory(app1)
add_subdirectory(app2)
add_subdirectory(app3)

add_library(ShMemLib STATIC lib.cpp) # or SHARED

# Make sure header files are findable by other targets
target_include_directories(lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link the OpenCV and Boost dependencies to the library
target_link_libraries(ShMemLib PRIVATE
    OpenCV::opencv
    SQLite3::SQLite3
    Boost::interprocess
)


find_package(Boost REQUIRED COMPONENTS system filesystem)





# project_root/lib/CMakeLists.txt
add_library(MyInterprocessLibrary STATIC lib.cpp) # You can use STATIC or SHARED

# Make the 'lib/' directory publicly available as an include path for targets that link to MyLib
target_include_directories(MyInterprocessLibrary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Find Boost.Interprocess
find_package(Boost COMPONENTS Interprocess REQUIRED)

# Link Boost to the library
target_link_libraries(MyInterprocessLibrary PRIVATE Boost::interprocess)













# cmake_minimum_required(VERSION 3.10)
# project(MyInterprocessLibrary CXX)

# find_package(Boost 1.82 COMPONENTS system interprocess REQUIRED)

# add_library(MyInterprocessLibrary STATIC
#     src/my_interprocess_class.cpp
#     src/another_file.cpp
# )

# if (Boost_FOUND)
#     target_include_directories(MyInterprocessLibrary PUBLIC ${Boost_INCLUDE_DIRS})
#     target_link_libraries(MyInterprocessLibrary PUBLIC Boost::system Boost::interprocess)
# endif()
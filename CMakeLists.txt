cmake_minimum_required(VERSION 2.8)
project(KCFCpp)

find_package(OpenCV 3.2 REQUIRED)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
add_executable( KCF ${sourcefiles} )
target_link_libraries( KCF ${OpenCV_LIBS})
set(CMAKE_VERBOSE_MAKEFILE on)





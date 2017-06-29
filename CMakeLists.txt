cmake_minimum_required(VERSION 2.8)
project(KCFCpp)

find_package(OpenCV 3.2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
add_executable( KCF src/fhog.cpp src/kcftracker.cpp src/runtracker.cpp )
target_link_libraries( KCF ${OpenCV_LIBS})
add_executable( KCF_gby src/fhog.cpp src/kcftracker.cpp src/runtracker_gby.cpp )
target_link_libraries( KCF_gby ${OpenCV_LIBS} ${Boost_LIBRARIES})
set(CMAKE_VERBOSE_MAKEFILE on)





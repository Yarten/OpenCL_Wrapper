cmake_minimum_required(VERSION 2.8)

project(OpenCL)

set(CMAKE_CXX_FLAGS "-std=c++11")
add_compile_options(-fpermissive)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

list(APPEND CMAKE_PREFIX_PATH "/media/yarten/YARTEN/Packages/Linux")
find_package(boost)
find_package(cuda9)

include_directories(
        ${boost_INCLUDE_DIRS}
        ${cuda9_INCLUDE_DIRS}
        include
)

add_executable(${PROJECT_NAME}
        src/OpenCL.cpp include/OpenCL.h src/main.cpp include/Buffer.h include/Event.h src/Event.cpp)

target_link_libraries(${PROJECT_NAME} ${boost_LIBS} ${cuda9_LIBS})

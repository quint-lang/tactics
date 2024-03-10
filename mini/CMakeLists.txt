set(MINI_USE_GFLAGS ${USE_GFLAGS}) # used in cmake_macros.h.in
set(MINI_USE_GLOG ${USE_GLOG}) # used in cmake_macros.h.in
set(MINI_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
set(MINI_USE_NUMA ${USE_NUMA})
set(MINI_USE_MSVC_STATIC_RUNTIME ${CAFFE2_USE_MSVC_STATIC_RUNTIME})
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/macros/cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/mini/macros/cmake_macros.h)

file(GLOB MINI_SRCS
        *.cpp
        core/*.cpp
        macros/*.cpp
        util/*.cpp
)

file(GLOB MINI_HEADERS
        *.h
        core/*.h
        macros/*.h
        util/*.h
)

add_library(mini ${MINI_SRCS} ${MINI_HEADERS})

target_include_directories(mini PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

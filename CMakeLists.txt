cmake_minimum_required(VERSION 3.10)

project(tactics)

if(POLICY CMP007)
    cmake_policy(SET CMP007 NEW)
endif()

if(POLICY CMP006)
    cmake_policy(SET CMP006 NEW)
endif()

option(BUILD_NO_TEST "Do not build tactics test binaries" OFF)
option(BUILD_BINARY "Build C++ binaries" OFF)
option(BUILD_DOCS "Build Tactics documentation" OFF)
option(BUILD_CUSTOM_PROTOBUF "Build and use tactics' own protobuf under third_party" ON)
option(BUILD_PYTHON "Build Python binaries" ON)
option(BUILD_SHARED_LIBS "Build libtactics.so" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT LLVM_LIBRARY_DIR)
    if(WIN32)
        set(LLVM_DIR "E://workspace//open-source//lib//llvm//lib//cmake//llvm")
        find_package(LLVM 17 REQUIRED COMPONENTS nvptx amdgpu)

        include_directories(${LLVM_INCLUDE_DIRS})
        separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
        add_definitions(${LLVM_DEFINITIONS_LIST})

        llvm_map_components_to_libnames(LLVM_LIBRARIES support core
                NVPTXInfo nvptxcodegen
                AMDGPUInfo AMDGPUcodegen
        )
    else()
        find_package(LLVM 11 REQUIRED COMPONENTS "nvptx;amdgpu")
    endif()

    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

    # FindLLVM outputs LLVM_LIBRARY_DIRS but we expect LLVM_LIBRARY_DIR here
    set(LLVM_LIBRARY_DIR ${LLVM_LIBRARY_DIRS})

    if(APPLE)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14")
    endif()
    set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
# sometimes we don't want to use llvm-config, since it may have been downloaded for some specific linux distros
else ()
    set(LLVM_LDFLAGS "-L${LLVM_LIBRARY_DIR}")
    set(LLVM_LIBRARIES
        LLVMNVPTXCodeGen
        LLVMNVPTXDesc
        LLVMNVPTXInfo
        LLVMAMDGPUDisassembler
        LLVMMCDisassembler
        LLVMAMDGPUCodeGen
        LLVMMIRParser
        LLVMGlobalISel
        LLVMSelectionDAG
        LLVMipo
        LLVMInstrumentation
        LLVMVectorize
        LLVMLinker
        LLVMIRReader
        LLVMAsmParser
        LLVMFrontendOpenMP
        LLVMAsmPrinter
        LLVMDebugInfoDWARF
        LLVMCodeGen
        LLVMTarget
        LLVMScalarOpts
        LLVMInstCombine
        LLVMAggressiveInstCombine
        LLVMTransformUtils
        LLVMBitWriter
        LLVMAnalysis
        LLVMProfileData
        LLVMObject
        LLVMTextAPI
        LLVMBitReader
        LLVMAMDGPUAsmParser
        LLVMMCParser
        LLVMAMDGPUDesc
        LLVMAMDGPUUtils
        LLVMMC
        LLVMDebugInfoCodeView
        LLVMDebugInfoMSF
        LLVMCore
        LLVMRemarks
        LLVMBitstreamReader
        LLVMBinaryFormat
        LLVMAMDGPUInfo
        LLVMSupport
        LLVMDemangle
        LLVMPasses
        LLVMAnalysis
        LLVMTransformUtils
        LLVMScalarOpts
        LLVMTransformUtils
        LLVMipo
        LLVMObjCARCOpts
        LLVMCoroutines
        LLVMAnalysis
        )
    set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})
if (MLIR_FOUND)
    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
        NO_DEFAULT_PATH)

    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

    include(TableGen) # required by AddMLIR
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)

    # Disable warnings that show up in external code (gtest;pybind11)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default")

    include_directories(${MLIR_INCLUDE_DIRS})
    include_directories(${LLVM_INCLUDE_DIRS})
    include_directories(${PROJECT_SOURCE_DIR}/include)
    include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files
endif ()

add_subdirectory(mini)
add_subdirectory(tactics)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)


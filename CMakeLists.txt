cmake_minimum_required(VERSION 3.2)

set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(HEonGPU LANGUAGES C CXX CUDA ASM)

if(NOT DEFINED MODULAR_REDUCTION_TYPE)
    set(MODULAR_REDUCTION_TYPE 0)
endif()



# Google Test'i projenize çekmek için FetchContent modülünü kullanın
include(FetchContent)

# Google Test kaynak kodlarını projenin alt dizinine çekin
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.0
  SOURCE_DIR     "${CMAKE_BINARY_DIR}/googletest-src" # Buradaki konumu değiştirebilirsiniz
)

# Google Test'i projeye ekleyin
FetchContent_MakeAvailable(googletest)

# SEAL
FetchContent_Declare(
  seal
  GIT_REPOSITORY https://github.com/microsoft/SEAL.git
  GIT_TAG        master  # or the desired version/tag
)
FetchContent_MakeAvailable(seal)



#set(HE_SOURCE_FILES "")
add_subdirectory(native/src)
add_library(HEonGPU STATIC ${HE_SOURCE_FILES})


if(NOT "${CMAKE_CUDA_BYTE_ORDER}" STREQUAL "LITTLE_ENDIAN")
    message(FATAL_ERROR "Target architecture must be little-endian")
endif()

set_target_properties(HEonGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>")


add_executable(HE_test1 native/test/HE_test1.cu)
add_executable(HE_test2 native/test/HE_test2.cu)

add_executable(HE_bench1 native/bench/HE_bench1.cu)



if(MODULAR_REDUCTION_TYPE EQUAL 0)
    add_compile_definitions(BARRETT_64)
    target_compile_definitions(HE_test1 PRIVATE BARRETT_64)
    target_compile_definitions(HE_test2 PRIVATE BARRETT_64)
    target_compile_definitions(HE_bench1 PRIVATE BARRETT_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 1)
    add_compile_definitions(GOLDILOCKS_64)
    target_compile_definitions(HE_test1 PRIVATE GOLDILOCKS_64)
    target_compile_definitions(HE_test2 PRIVATE GOLDILOCKS_64)
    target_compile_definitions(HE_bench1 PRIVATE GOLDILOCKS_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 2)
    add_compile_definitions(PLANTARD_64)
    target_compile_definitions(HE_test1 PRIVATE PLANTARD_64)
    target_compile_definitions(HE_test2 PRIVATE PLANTARD_64)
    target_compile_definitions(HE_bench1 PRIVATE PLANTARD_64)
else()
    message(FATAL_ERROR "Please use valid MODULAR_REDUCTION_TYPE! (0,1,2)")
endif()


target_link_libraries(HE_test1 PRIVATE HEonGPU gtest_main SEAL::seal)
set_target_properties(HE_test1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(HE_test2 PRIVATE HEonGPU gtest_main SEAL::seal)
set_target_properties(HE_test2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(HE_bench1 PRIVATE HEonGPU)
set_target_properties(HE_bench1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
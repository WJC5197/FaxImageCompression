# project name
project(FaxImageCompress C CXX)
# cmake version
cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER             "/usr/sbin/clang")
set(CMAKE_C_FLAGS                "-Wall -std=c17")
set(CMAKE_C_FLAGS_DEBUG          "-g")
set(CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

set(CMAKE_CXX_COMPILER             "/usr/sbin/clang++")
set(CMAKE_CXX_FLAGS                "-Wall -std=c++20")
set(CMAKE_CXX_FLAGS_DEBUG          "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

set(CMAKE_AR      "/usr/sbin/llvm-ar")
set(CMAKE_LINKER  "/usr/sbin/llvm-ld")
set(CMAKE_NM      "/usr/sbin/llvm-nm")
set(CMAKE_RANLIB  "/usr/sbin/llvm-ranlib")
set(CMAKE_OBJDUMP "/usr/sbin/llvm-objdump")


# find OpenCV
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# show the message of OpenCV
message(STATUS "OpenCV library status:")
message(STATUS "    version: 	${OpenCV_VERSION}")
message(STATUS "    headers: 	${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: 	${OpenCV_LIBS}")

# link headers
include_directories({OpenCV_INCLUDE_DIRS})


# 添加源代码文件到SRC_LIST变量中
# aux_source_directory(./src SRC_LIST)
aux_source_directory(./test TEST_LIST)

file(GLOB HEADERS ./*.hpp)
source_group("Headers" FILES ${HEADERS})
add_library(Util ${HEADERS})
set_target_properties(Util PROPERTIES LINKER_LANGUAGE CXX)

# Main
    # 生成可执行文件
add_executable(Main ./main.cpp)
target_compile_options(Main PUBLIC ${OpenMP_CXX_FLAGS})
target_link_libraries(Main PUBLIC
    ${OpenCV_LIBS}
    ${OpenMP_CXX_FLAGS}
    OpenMP::OpenMP_CXX)

# img test
add_executable(Test ./test.cpp)
target_link_libraries(Test ${OpenCV_LIBS})
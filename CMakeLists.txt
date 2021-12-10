project(EasyCTR)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_C_COMPILER "/usr/local/bin/gcc")
SET(CMAKE_CXX_COMPILER "/usr/local/bin/g++")

# tf compile and link flags
exec_program(
    python
    ARGS "-c \"import tensorflow as tf; print(\\\" \\\".join(tf.sysconfig.get_compile_flags()))\""
    OUTPUT_VARIABLE TF_CFLAGS)
exec_program(
    python
    ARGS
    "-c \"import tensorflow as tf; print(\\\" \\\".join(tf.sysconfig.get_link_flags()))\""
    OUTPUT_VARIABLE TF_LFLAGS)

# c flags
set(CMAKE_C_FLAGS "-std=gnu99 -O2 -Wall -Wextra -DLUA_COMPAT_5_2")
# cxx flags
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall -Werror -Wno-deprecated -Wno-unused-function ${TF_CFLAGS}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Linker flags
set(CMAKE_SHARED_LINKER_FLAGS "${TF_LFLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${TF_LFLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

add_subdirectory(deps)
add_subdirectory(assembler)
add_subdirectory(tools)
add_subdirectory(ops)
add_subdirectory(test)

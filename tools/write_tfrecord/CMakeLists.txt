
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
add_executable(
    write_tfrecord
    ${SRCS}
)
target_link_libraries(write_tfrecord assembler tensorflow_framework)

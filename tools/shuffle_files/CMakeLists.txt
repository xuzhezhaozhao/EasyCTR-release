
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
add_executable(
    shuffle_files
    ${SRCS}
)
target_link_libraries(shuffle_files)

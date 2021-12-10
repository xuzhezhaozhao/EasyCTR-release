
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
add_executable(
    string_indexer
    ${SRCS}
)
target_link_libraries(string_indexer assembler tensorflow_framework)

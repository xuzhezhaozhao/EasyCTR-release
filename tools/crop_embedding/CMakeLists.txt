
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
add_executable(
    crop_embedding
    ${SRCS}
)
target_link_libraries(crop_embedding assembler tensorflow_framework)

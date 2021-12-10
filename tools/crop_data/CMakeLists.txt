
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
add_executable(
    crop_data
    ${SRCS}
)
target_link_libraries(crop_data assembler tensorflow_framework)

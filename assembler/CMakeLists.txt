
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
set(ASSEMBLER_SRCS ${SRCS} CACHE INTERNAL "assembler src")
add_library(
    assembler
    ${SRCS}
)
target_link_libraries(assembler jsoncpp attr murmurhash3 lua)

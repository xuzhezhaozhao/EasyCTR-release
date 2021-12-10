
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
set(MURMURHASH3_SRCS ${SRCS} CACHE INTERNAL "murmurhash3 src")
add_library(
    murmurhash3
    ${SRCS}
)

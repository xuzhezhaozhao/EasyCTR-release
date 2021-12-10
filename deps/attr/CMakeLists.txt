
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
set(ATTR_SRCS ${SRCS} CACHE INTERNAL "attr src")
add_library(
    attr
    ${SRCS}
)

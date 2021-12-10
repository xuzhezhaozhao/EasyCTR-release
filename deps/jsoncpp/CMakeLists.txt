
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
set(JSONCPP_SRCS ${SRCS} CACHE INTERNAL "jsoncpp src")
add_library(
    jsoncpp
    ${SRCS}
)

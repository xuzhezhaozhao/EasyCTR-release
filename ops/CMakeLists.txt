
add_library(
    assembler_ops SHARED
    assembler_kernels.cc
    assembler_ops.cc
    ${JSONCPP_SRCS}
    ${ATTR_SRCS}
    ${MURMURHASH3_SRCS}
    ${ASSEMBLER_SRCS}
    ${LUA_SRCS}
)
target_link_libraries(assembler_ops tensorflow_framework)

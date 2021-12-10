
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "lua",
    visibility = ["//visibility:public"],
    srcs = glob(
        ["deps/lua/src/*.h", "deps/lua/src/*.c"],
        exclude=["deps/lua/src/lua.c", "deps/lua/src/luac.c"]
    ),
    deps = [
        "//tensorflow/core:framework",
    ],
    copts = ["-std=gnu99", "-O2", "-DLUA_COMPAT_5_2"],
    linkopts = ["-ldl"],
    includes = ["./"],
)

cc_library(
    name = "assembler_ops",
    visibility = ["//visibility:public"],
    srcs = glob(["assembler/*.h",
                 "assembler/*.cpp",
                 "assembler/*.c",
                 "deps/jsoncpp/*.cpp",
                 "deps/jsoncpp/json/*.h",
                 "deps/attr/*.c",
                 "deps/attr/*.h",
                 "deps/murmurhash3/*.h",
                 "deps/murmurhash3/*.cpp",
                 "ops/*.cc",
                 "deps/lua/src/*.h",
                 ]),
    deps = ["@org_tensorflow//tensorflow/core:framework_headers_lib",
            "//tensorflow/easyctr:lua",
    ],
    includes = ["./"],
)

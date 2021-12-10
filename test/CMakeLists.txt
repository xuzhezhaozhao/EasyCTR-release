
add_executable(
    test_conf_parser
    test_conf_parser.cpp
)
target_link_libraries(test_conf_parser assembler jsoncpp tensorflow_framework)


add_executable(
    test_assembler
    test_assembler.cpp
)
target_link_libraries(test_assembler assembler jsoncpp tensorflow_framework)


add_executable(
    test_scheme
    test_scheme.cpp
)
target_link_libraries(test_scheme assembler jsoncpp tensorflow_framework)


add_executable(
    test_assembler_serialize
    test_assembler_serialize.cpp
)
target_link_libraries(test_assembler_serialize assembler jsoncpp tensorflow_framework)

add_library(
    test_random_access_file SHARED
    test_random_access_file.cc
)
target_link_libraries(test_random_access_file tensorflow_framework)

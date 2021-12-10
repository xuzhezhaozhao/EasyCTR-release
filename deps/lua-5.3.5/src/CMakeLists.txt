
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lua.c)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/luac.c)

set(LUA_SRCS ${SRCS} CACHE INTERNAL "lua src")
add_library(
    lua
    ${SRCS}
)
target_link_libraries(lua dl)

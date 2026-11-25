set(TARGET llama-mem)

add_executable(${TARGET} main.cpp)

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(${TARGET} PRIVATE mem llama ggml)

install(TARGETS ${TARGET} RUNTIME)

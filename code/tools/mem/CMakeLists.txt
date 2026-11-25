set(TARGET mem)

add_library(${TARGET} STATIC
    mem.cpp
    mem-loader.cpp
    mem-encoder.cpp
    mem-retrieve.cpp
    mem-card.cpp
    mem-journal.cpp
    mem-dream.cpp
    mem.h
    mem-internal.h
    mem-card.h
    mem-journal.h
    mem-dream.h
)

target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})

target_compile_features(${TARGET} PUBLIC cxx_std_17)

target_link_libraries(${TARGET} PUBLIC llama ggml vendor::nlohmann)

if (LLAMA_BUILD_TOOLS)
    add_subdirectory(mem-cli)
endif()

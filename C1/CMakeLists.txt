cmake_minimum_required(VERSION 3.29)
project(C1)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

add_executable(C1 main.cpp
        Shader.h)

target_link_libraries(C1
        glfw
        glad::glad
        OpenGL::GL
)

file(COPY vertex.glsl DESTINATION ${CMAKE_BINARY_DIR})
file(COPY fragment.glsl DESTINATION ${CMAKE_BINARY_DIR})

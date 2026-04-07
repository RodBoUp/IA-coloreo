cmake_minimum_required(VERSION 3.29)
project(Tarea_Pathfindings)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

add_executable(Tarea_Pathfindings main.cpp
        Shader.h
        stb_easy_font.h)

target_link_libraries(Tarea_Pathfindings
        glfw
        glad::glad
        OpenGL::GL
)

file(COPY vertex.glsl DESTINATION ${CMAKE_BINARY_DIR})
file(COPY fragment.glsl DESTINATION ${CMAKE_BINARY_DIR})

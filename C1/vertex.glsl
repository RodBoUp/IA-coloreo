#version 330 core
layout (location = 0) in vec2 aPos;
uniform float zoom;
uniform vec2 camera;
uniform float aspect;

void main() {
    vec2 pos = (aPos - camera) * zoom;
    pos.x *= aspect;
    gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
}
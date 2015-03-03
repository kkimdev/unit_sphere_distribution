#version 150 core

uniform mat4 u_model_view_proj;

in vec3 a_pos;
in vec4 color;

out vec4 colorV;

void main() {
    colorV = color;
    gl_Position = u_model_view_proj * vec4(a_pos, 1.0);
}

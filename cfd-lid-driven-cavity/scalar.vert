#version 330 core
layout(location = 0) in vec2 aPos;     // (x, y)
layout(location = 1) in vec3 aColor;   // (r, g, b)

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec3 fragColor; // passed to fragment shader

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 0.0, 1.0);
    fragColor = aColor;
}
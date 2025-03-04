#version 330 core
// Input attributes: a 2D position and a 3-component side property.
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aSideProp;

// Pass the side property to the fragment shader.
out vec3 vSideProp;

// Uniform transformation matrices (optional, if you need transformations).
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Transform the vertex position. If you don't need transformations, you can simply do:
    // gl_Position = vec4(aPos, 0.0, 1.0);
    gl_Position = projection * view * model * vec4(aPos, 0.0, 1.0);
    
    // Pass along the side property (for example, a unique color per side).
    vSideProp = aSideProp;
}

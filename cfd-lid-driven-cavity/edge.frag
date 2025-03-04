#version 330 core
// Receive the side property (color) from the vertex shader.
in vec3 vSideProp;

// Output color.
out vec4 FragColor;

void main()
{
    // Use the side property as the color.
    FragColor = vec4(vSideProp, 1.0);
}

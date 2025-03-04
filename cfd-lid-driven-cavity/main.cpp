#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include "shader_configure.h"

int main(void)
{
    // Initialize GLFW.
    if (!glfwInit())
        return -1;

    // Create a windowed mode window and its OpenGL context.
    GLFWwindow* window = glfwCreateWindow(1800 , 1800, "Square Example", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW.
    if (glewInit() != GLEW_OK)
    {
        std::cout << "Error initializing GLEW\n";
        return -1;
    }

    // Create and compile the shader once.
    Shader edgeShader("edge.vert", "edge.frag");

    // Set up an orthographic projection (world units in meters).
    // For instance, we use a view spanning -5 to 5 in both x and y.
    glm::mat4 projection = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, -1.0f, 1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 model = glm::mat4(1.0f);
    edgeShader.use();
    edgeShader.setMat4("projection", projection);
    edgeShader.setMat4("view", view);
    edgeShader.setMat4("model", model);

    // Set attribute pointer for vertex positions (location = 0).
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    GLuint topEdgeVAO, topEdgeVBO;
    glGenVertexArrays(1, &topEdgeVAO);
    glGenBuffers(1, &topEdgeVBO);

    glBindVertexArray(topEdgeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, topEdgeVBO);
    float topEdge[] = { -1.0f,  1.0f,  1.0f,  1.0f };
    glBufferData(GL_ARRAY_BUFFER, sizeof(topEdge), topEdge, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    GLuint rightEdgeVAO, rightEdgeVBO;
    glGenVertexArrays(1, &rightEdgeVAO);
    glGenBuffers(1, &rightEdgeVBO);

    glBindVertexArray(rightEdgeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, rightEdgeVBO);
    float rightEdge[] = { 1.0f,  1.0f,  1.0f, -1.0f };
    glBufferData(GL_ARRAY_BUFFER, sizeof(rightEdge), rightEdge, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    GLuint bottomEdgeVAO, bottomEdgeVBO;
    glGenVertexArrays(1, &bottomEdgeVAO);
    glGenBuffers(1, &bottomEdgeVBO);

    glBindVertexArray(bottomEdgeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, bottomEdgeVBO);
    float bottomEdge[] = { 1.0f, -1.0f, -1.0f, -1.0f };
    glBufferData(GL_ARRAY_BUFFER, sizeof(bottomEdge), bottomEdge, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    GLuint leftEdgeVAO, leftEdgeVBO;
    glGenVertexArrays(1, &leftEdgeVAO);
    glGenBuffers(1, &leftEdgeVBO);

    glBindVertexArray(leftEdgeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, leftEdgeVBO);
    float leftEdge[] = { -1.0f, -1.0f, -1.0f,  1.0f };
    glBufferData(GL_ARRAY_BUFFER, sizeof(leftEdge), leftEdge, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Main render loop.
    while (!glfwWindowShouldClose(window))
    {
        // Clear the screen.
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Use our shader.
        edgeShader.use();

        // Set the uniform color for the edge (red in this case).
        GLint edgeColorLoc = glGetUniformLocation(edgeShader.ID, "edgeColor");
        glUniform3f(edgeColorLoc, 1.0f, 0.0f, 0.0f);

        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);
        // Bind the top edge VAO and draw the line.
        glBindVertexArray(topEdgeVAO);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);

        // Bind the right edge VAO and draw the line.
        glBindVertexArray(rightEdgeVAO);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);

        // Bind the bottom edge VAO and draw the line.
        glBindVertexArray(bottomEdgeVAO);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);

        // Bind the left edge VAO and draw the line.
        glBindVertexArray(leftEdgeVAO);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);


        glLineWidth(2.0f);



        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup.
    glDeleteVertexArrays(1, &topEdgeVAO);
    glDeleteBuffers(1, &topEdgeVBO);
    glfwTerminate();
    return 0;
}

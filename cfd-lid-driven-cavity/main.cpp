// OpenGL libraries
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// C++ libraries
#include <iostream>
#include <vector>
#include <cmath>

// Custom headers
#include "shader_configure.h"

//--------------------------------------------------
// Global CFD parameters
//--------------------------------------------------
static const int Nx = 99;            // Number of cells in x-direction
static const int Ny = 99;            // Number of cells in y-direction
static const float Lx = 1.0f;        // Domain size in x
static const float Ly = 1.0f;        // Domain size in y
static const float Re = 100.0f;      // Reynolds number (example)
static const float dt = 0.001f;      // Time step size
static const int  MAX_ITER = 10000;  // Maximum number of SIMPLE iterations
static const float TOL = 1e-5f;      // Tolerance for convergence

// Arrays to store the velocity and pressure fields
static std::vector<float> u; // size: (Nx+1)*Ny
static std::vector<float> v; // size: Nx*(Ny+1)
static std::vector<float> p; // size: Nx*Ny

// Helper to index into 2D arrays in row-major order
inline int idxU(int i, int j) {
    // i in [0..Nx], j in [0..Ny-1]
    return j * (Nx + 1) + i;
}
inline int idxV(int i, int j) {
    // i in [0..Nx-1], j in [0..Ny]
    return j * Nx + i;
}
inline int idxP(int i, int j) {
    // i in [0..Nx-1], j in [0..Ny-1]
    return j * Nx + i;
}

// Grid spacing
static float dx = Lx / Nx;
static float dy = Ly / Ny;

//--------------------------------------------------
// OpenGL Stuff
//--------------------------------------------------

GLFWwindow* window;
GLuint scalarVAO = 0, scalarVBO = 0, scalarEBO = 0;
bool scalarFieldInitialized = false;

//--------------------------------------------------
// CFD Function Declarations
//--------------------------------------------------
void initializeFields();
void applyBoundaryConditions();
void momentumEquationsPredict(std::vector<float>& uStar, std::vector<float>& vStar);
void pressureCorrection(std::vector<float>& pPrime, const std::vector<float>& uStar, const std::vector<float>& vStar);
void updateFields(const std::vector<float>& pPrime, std::vector<float>& uStar, std::vector<float>& vStar);
float computeContinuityResidual();

//--------------------------------------------------
// OpenGL Helper Functions ( Visualization )
//--------------------------------------------------
void createEdges(GLuint& VAO, GLuint& VBO, const float* data, int size);
void renderEdges(GLuint VAO, int numPoints);
void renderVelocityVectors();
static void getColormapRGB(float val, float& r, float& g, float& b);
void renderScalarField(Shader& scalarShader, const glm::mat4& projection,
                       const glm::mat4& view, const glm::mat4& model);
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // Update the viewport to cover the new window dimensions.
    glViewport(0, 0, width, height);
}

glm::mat4 windowCentered(GLFWwindow* window);
//--------------------------------------------------
// main
//--------------------------------------------------
int main(void)
{
    // ------------------------------------------------
    // Initialize Boilerplate OpenGL/GLFW Window
    // ------------------------------------------------
    if (!glfwInit())
        return -1;

    window = glfwCreateWindow(1000, 1000, "Lid-Driven Cavity (SIMPLE + OpenGL)", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // for resizing

    if (glewInit() != GLEW_OK)
    {
        std::cout << "Error initializing GLEW\n";
        return -1;
    }

    // ------------------------------------------------
	// Compile and Link Shaders with shader_configure.h
    // ------------------------------------------------
    Shader edgeShader("edge.vert", "edge.frag");
    Shader scalarShader("scalar.vert", "scalar.frag");
    Shader velocityShader("velocity.vert", "velocity.frag");
    glm::mat4 projection = glm::ortho(-0.1f, 1.1f, -0.1f, 1.1f, -1.0f, 1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 model = glm::mat4(1.0f);
    edgeShader.use();
    edgeShader.setMat4("projection", projection);
    edgeShader.setMat4("view", view);
    edgeShader.setMat4("model", model);

    // ------------------------------------------------
    // Create geometry for the square boundary
    // ------------------------------------------------
    // Square edges: top, right, bottom, left

                        //
	float ext = 0.0025; // HACK: Extend the edges slightly to avoid gaps in rendering
                        // Should be fixed by connecting the lines, or by drawing squares in the corners
                        //

    float topEdge[] = { 0.0f - ext, 1.0f,  1.0f + ext, 1.0f };
    float rightEdge[] = { 1.0f, 1.0f + ext,  1.0f, 0.0f - ext};
    float bottomEdge[] = { 1.0f + ext, 0.0f,  0.0f - ext, 0.0f };
    float leftEdge[] = { 0.0f, 0.0f - ext,  0.0f, 1.0f + ext};

    GLuint topEdgeVAO, topEdgeVBO;
    createEdges(topEdgeVAO, topEdgeVBO, topEdge, 4);
    GLuint rightEdgeVAO, rightEdgeVBO;
    createEdges(rightEdgeVAO, rightEdgeVBO, rightEdge, 4);
    GLuint bottomEdgeVAO, bottomEdgeVBO;
    createEdges(bottomEdgeVAO, bottomEdgeVBO, bottomEdge, 4);
    GLuint leftEdgeVAO, leftEdgeVBO;
    createEdges(leftEdgeVAO, leftEdgeVBO, leftEdge, 4);

    // ------------------------------------------------
    // Allocate & Initialize CFD fields
    // ------------------------------------------------
    u.resize((Nx + 1) * Ny, 0.0f);
    v.resize(Nx * (Ny + 1), 0.0f);
    p.resize(Nx * Ny, 0.0f);

    initializeFields();

    // ------------------------------------------------
    // Main Simulation Loop (SIMPLE Algorithm)
    // ------------------------------------------------
    int iteration = 0;
    float finalResidual = 1.0f;

    while (!glfwWindowShouldClose(window) && iteration < MAX_ITER && finalResidual > TOL)
    {
        glm::mat4 projection = windowCentered(window);

        // Clear screen
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Apply Boundary Conditions
        applyBoundaryConditions();

        // Momentum Equations to get provisional velocities (uStar, vStar)
        static std::vector<float> uStar((Nx + 1) * Ny, 0.0f);
        static std::vector<float> vStar(Nx * (Ny + 1), 0.0f);

        momentumEquationsPredict(uStar, vStar);

        // Solve Pressure Correction p' from Poisson equation
        static std::vector<float> pPrime(Nx * Ny, 0.0f);
        pressureCorrection(pPrime, uStar, vStar);

        // Update p, u, v with correction
        updateFields(pPrime, uStar, vStar);

        // Check continuity residual
        finalResidual = computeContinuityResidual();
        if (iteration % 100 == 0) {
            std::cout << "Iteration: " << iteration << ", Residual: " << finalResidual << std::endl;
        }
        
        if (finalResidual < TOL) {
            std::cout << "Converged at iteration = " << iteration << " with residual = " << finalResidual << "\n";
        }

        iteration++;

        // ---------------------------------------------------
        // VISUALIZATION - Render the domain & velocity field
        // ---------------------------------------------------
        edgeShader.use();
        GLint edgeColorLoc = glGetUniformLocation(edgeShader.ID, "edgeColor");

		// Draw square boundary edges
        glUniform3f(edgeColorLoc, 0.0f, 0.0f, 0.0f); // black
        renderEdges(topEdgeVAO, 2);
        renderEdges(rightEdgeVAO, 2);
        renderEdges(bottomEdgeVAO, 2);
        renderEdges(leftEdgeVAO, 2);

        // Render the background (colored by velocity magnitude)
        scalarShader.use();
        scalarShader.setMat4("projection", projection);
        scalarShader.setMat4("view", view);
        scalarShader.setMat4("model", model);
        
        renderScalarField(scalarShader, projection, view, model);

        // Render the boundary edges
        edgeShader.use();
        edgeShader.setMat4("projection", projection);
        edgeShader.setMat4("view", view);
        edgeShader.setMat4("model", model);
        
        // Render velocity vectors in white
        velocityShader.use();
        velocityShader.setMat4("projection", projection);
        velocityShader.setMat4("view", view);
        velocityShader.setMat4("model", model);
        GLint arrowColorLoc = glGetUniformLocation(velocityShader.ID, "arrowColor");
        glUniform3f(arrowColorLoc, 1.0f, 1.0f, 1.0f);
		
        renderVelocityVectors();

        // OpenGL Commands
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //
    // FIXME: Final residual takes to long to reach, likely an issue with the convergence criteria.
    //
    if (iteration >= MAX_ITER) {
        std::cout << "Reached maximum iterations: " << MAX_ITER << " with residual: " << finalResidual << std::endl;
    }

    // OpenGL Cleanup
    glDeleteVertexArrays(1, &topEdgeVAO);
    glDeleteBuffers(1, &topEdgeVBO);
    glDeleteVertexArrays(1, &rightEdgeVAO);
    glDeleteBuffers(1, &rightEdgeVBO);
    glDeleteVertexArrays(1, &bottomEdgeVAO);
    glDeleteBuffers(1, &bottomEdgeVBO);
    glDeleteVertexArrays(1, &leftEdgeVAO);
    glDeleteBuffers(1, &leftEdgeVBO);

    glfwTerminate();
    return 0;
}

//--------------------------------------------------
// Function Definitions
//--------------------------------------------------

// Initialize Values (u,v,p = 0) and (u = 1 at top boundary)
void initializeFields()
{
    // Initialize all values to zero
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i <= Nx; i++) {
            u[idxU(i, j)] = 0.0f;
        }
    }
    
    for (int j = 0; j <= Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            v[idxV(i, j)] = 0.0f;
        }
    }
    
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            p[idxP(i, j)] = 0.0f;
        }
    }
    
    // Set lid velocity (top boundary)
    for (int i = 0; i <= Nx; i++) {
        u[idxU(i, Ny-1)] = 1.0f;
    }
}

// Apply the boundary conditions for lid-driven CFD
void applyBoundaryConditions()
{
    // Top boundary (lid): u = 1.0, v = 0
    for (int i = 0; i <= Nx; i++) {
        u[idxU(i, Ny-1)] = 1.0f;  // Lid velocity
    }
    
    for (int i = 0; i < Nx; i++) {
        v[idxV(i, Ny)] = 0.0f;    // No flow through top boundary
    }

    // Bottom boundary: u = 0, v = 0
    for (int i = 0; i <= Nx; i++) {
        u[idxU(i, 0)] = 0.0f;     // No-slip at bottom wall
    }
    
    for (int i = 0; i < Nx; i++) {
        v[idxV(i, 0)] = 0.0f;     // No flow through bottom boundary
    }

    // Left boundary: u = 0, v = 0
    for (int j = 0; j < Ny; j++) {
        u[idxU(0, j)] = 0.0f;     // No-slip at left wall
    }
    
    for (int j = 0; j <= Ny; j++) {
        v[idxV(0, j)] = 0.0f;  // No flow through left boundary
    }

    // Right boundary: u = 0, v = 0
    for (int j = 0; j < Ny; j++) {
        u[idxU(Nx, j)] = 0.0f;    // No-slip at right wall
    }
    
    for (int j = 0; j <= Ny; j++) {
        v[idxV(Nx-1, j)] = 0.0f;  // No flow through right boundary
    }

    // Set reference pressure point to avoid singularity
    p[idxP(0, 0)] = 0.0f;
}

// Solve the momentum equations
void momentumEquationsPredict(std::vector<float>& uStar, std::vector<float>& vStar)
{
    // Initialize uStar, vStar with current values
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i <= Nx; i++) {
            uStar[idxU(i, j)] = u[idxU(i, j)];
        }
    }
    
    for (int j = 0; j <= Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            vStar[idxV(i, j)] = v[idxV(i, j)];
        }
    }
    
    // Compute u-momentum for interior points
    for (int j = 1; j < Ny-1; j++) {
        for (int i = 1; i < Nx; i++) {
			// Get u at point and surrounding points for approximating partial differential equations
            float uC = u[idxU(i, j)];
            float uE = u[idxU(i+1, j)];
            float uW = u[idxU(i-1, j)];
            float uN = u[idxU(i, j+1)];
            float uS = u[idxU(i, j-1)];
            
            // Interpolate velocity at cell faces for convection
            float uFace = 0.5f * (uC + uE);
            float vFace = 0.5f * (v[idxV(i-1, j)] + v[idxV(i-1, j+1)]);
            
            /*
			* Why i-1 ? : We want the values that relate to u(i,j) for vFace, so we take the
			* left points (i-1,j) and (i-1,j+1) to compute the average. This is because
            * the v values to the left, are the values that directly match to the
            * u value, do not get confused with the fact that physically these nodes are half-a-node to the left of u(i,j).
            */

            // Compute convection term
            float convection = uFace * (uE - uC) / dx + vFace * (uN - uS) / (2.0f * dy);
            
            // Compute diffusion term
            float diffusion = (uE - 2.0f * uC + uW) / (dx * dx) + (uN - 2.0f * uC + uS) / (dy * dy);
            
            // Compute rate of pressure change across face
            float pE = p[idxP(i, j)];
            float pW = p[idxP(i-1, j)];
            float dpdx = (pE - pW) / dx;
            
            // Update uStar
            uStar[idxU(i, j)] = uC + dt * (-convection + diffusion / Re - dpdx);
        }
    }
    
    // Repeat previous steps for u-momentum but for v-momentum,
    // Compute v-momentum for interior points
    for (int j = 1; j < Ny; j++) {
        for (int i = 1; i < Nx-1; i++) {
            // Get v at point and surrounding points for approximating partial differential equations
            float vC = v[idxV(i, j)];
            float vE = v[idxV(i+1, j)];
            float vW = v[idxV(i-1, j)];
            float vN = v[idxV(i, j+1)];
            float vS = v[idxV(i, j-1)];
            
            // Interpolate velocity at cell faces for convection
            float uFace = 0.5f * (u[idxU(i, j-1)] + u[idxU(i+1, j-1)]);
            float vFace = 0.5f * (vC + vN);
            
            // Compute convection term
            float convection = uFace * (vE - vW) / (2.0f * dx) + vFace * (vN - vC) / dy;
            
            // Compute diffusion term
            float diffusion = (vE - 2.0f * vC + vW) / (dx * dx) + 
                              (vN - 2.0f * vC + vS) / (dy * dy);
            
            // Compute rate of pressure change across face
            float pN = p[idxP(i, j)];
            float pS = p[idxP(i, j-1)];
            float dpdy = (pN - pS) / dy;
            
            // Update vStar
            vStar[idxV(i, j)] = vC + dt * (-convection + diffusion / Re - dpdy);
        }
    }
    
    // Apply boundary conditions to uStar and vStar, these need to be applied each loop, hence why they're found here.
    // Top boundary (lid): u = 1.0, v = 0
    for (int i = 0; i <= Nx; i++) {
        uStar[idxU(i, Ny-1)] = 1.0f;
    }
    
    for (int i = 0; i < Nx; i++) {
        vStar[idxV(i, Ny)] = 0.0f;
    }

    // Bottom boundary: u = 0, v = 0
    for (int i = 0; i <= Nx; i++) {
        uStar[idxU(i, 0)] = 0.0f;
    }
    
    for (int i = 0; i < Nx; i++) {
        vStar[idxV(i, 0)] = 0.0f;
    }

    // Left boundary: u = 0, v = 0
    for (int j = 0; j < Ny; j++) {
        uStar[idxU(0, j)] = 0.0f;
    }
    
    for (int j = 0; j <= Ny; j++) {
        vStar[idxV(0, j)] = 0.0f;
    }

    // Right boundary: u = 0, v = 0
    for (int j = 0; j < Ny; j++) {
        uStar[idxU(Nx, j)] = 0.0f;
    }
    
    for (int j = 0; j <= Ny; j++) {
        vStar[idxV(Nx-1, j)] = 0.0f;
    }
}

// Solve the Poisson equation for pressure correction
void pressureCorrection(std::vector<float>& pPrime, const std::vector<float>& uStar, const std::vector<float>& vStar)
{
    // Initialize pPrime to zero
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            pPrime[idxP(i, j)] = 0.0f;
        }
    }
    
    // Setup for Poisson solver
    float relaxFactor = 1.7f;  // SOR relaxation factor (should be set to around 1.7-1.9 for 2D)
    int poissonMaxIters = 200; // Increase number of iterations for better convergence
	float poissonTol = 1e-4f;  // Determines when to stop iterations should it arrive at a certain convergence
    
    // Solve pressure correction equation with SOR
    for (int iter = 0; iter < poissonMaxIters; iter++) {
        float residualSum = 0.0f;
        
        for (int j = 1; j < Ny-1; j++) {
            for (int i = 1; i < Nx-1; i++) {
                // Calculate divergence at cell centers
                float uE = uStar[idxU(i+1, j)];
                float uW = uStar[idxU(i, j)];
                float vN = vStar[idxV(i, j+1)];
                float vS = vStar[idxV(i, j)];
                
                float divergence = (uE - uW) / dx + (vN - vS) / dy;
                
                // Initialize surrounding pressures
                float pE = pPrime[idxP(i+1, j)];
                float pW = pPrime[idxP(i-1, j)];
                float pN = pPrime[idxP(i, j+1)];
                float pS = pPrime[idxP(i, j-1)];
                
                // Calculate coefficient for central point
                float ap = 2.0f * (1.0f/(dx*dx) + 1.0f/(dy*dy));
                
                // Calculate new pressure using SOR
                float pNew = (1.0f - relaxFactor) * pPrime[idxP(i, j)] +
                             relaxFactor * ((pE + pW)/(dx*dx) + 
                             (pN + pS)/(dy*dy) - divergence/dt) / ap;
                
                // Calculate residual
                float residual = std::abs(pNew - pPrime[idxP(i, j)]);
                residualSum += residual;
                
                pPrime[idxP(i, j)] = pNew;
            }
        }
        
        // Apply boundary conditions for pressure correction
        // For pressure correction, use dp'/dn = 0 at boundaries
        
        // Left and right boundaries
        for (int j = 0; j < Ny; j++) {
            pPrime[idxP(0, j)] = pPrime[idxP(1, j)];         // Left: dp'/dx = 0
            pPrime[idxP(Nx-1, j)] = pPrime[idxP(Nx-2, j)];   // Right: dp'/dx = 0
        }
        
        // Bottom and top boundaries
        for (int i = 0; i < Nx; i++) {
            pPrime[idxP(i, 0)] = pPrime[idxP(i, 1)];         // Bottom: dp'/dy = 0
            pPrime[idxP(i, Ny-1)] = pPrime[idxP(i, Ny-2)];   // Top: dp'/dy = 0
        }
        
        // Check for convergence
        float avgResidual = residualSum / (Nx * Ny);
        if (avgResidual < poissonTol) {
            // std::cout << "Pressure correction converged in " << iter+1 << " iterations" << std::endl;
            break;
        }
    }
    
    // Ensure pressure correction has zero mean
    float pSum = 0.0f;
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            pSum += pPrime[idxP(i, j)];
        }
    }
    
    float pAvg = pSum / (Nx * Ny);
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            pPrime[idxP(i, j)] -= pAvg;
        }
    }
}

// Update p, u, v using p' corrections
void updateFields(const std::vector<float>& pPrime, std::vector<float>& uStar, std::vector<float>& vStar)
{
    // Update pressure field
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            p[idxP(i, j)] += pPrime[idxP(i, j)];
        }
    }
    
    // Update u velocity field
    for (int j = 0; j < Ny; j++) {
        for (int i = 1; i < Nx; i++) {
            float pL = pPrime[idxP(i-1, j)];
            float pR = pPrime[idxP(i, j)];
            float grad_p = (pR - pL) / dx;
            u[idxU(i, j)] = uStar[idxU(i, j)] - dt * grad_p;
        }
    }
    
    // Update v velocity field
    for (int j = 1; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            float pB = pPrime[idxP(i, j-1)];
            float pT = pPrime[idxP(i, j)];
            float grad_p = (pT - pB) / dy;
            v[idxV(i, j)] = vStar[idxV(i, j)] - dt * grad_p;
        }
    }
    
    // Apply boundary conditions
    applyBoundaryConditions();
}

// Compute the continuity residual
float computeContinuityResidual()
{
    float sumRes = 0.0f;
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            float uE = u[idxU(i+1, j)];
            float uW = u[idxU(i, j)];
            float vN = v[idxV(i, j+1)];
            float vS = v[idxV(i, j)];
            
            float divUV = (uE - uW) / dx + (vN - vS) / dy;
            sumRes += divUV * divUV;
        }
    }
    
    return std::sqrt(sumRes / (Nx * Ny));
}

//--------------------------------------------------
// OpenGL Function Definitions
//--------------------------------------------------

// Create edges for the domain
void createEdges(GLuint& VAO, GLuint& VBO, const float* data, int size)
{
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, size * sizeof(float), data, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Render edges of the domain
void renderEdges(GLuint VAO, int numPoints)
{
    glBindVertexArray(VAO);
    glLineWidth(10.0f);
    glDrawArrays(GL_LINES, 0, numPoints);
    glBindVertexArray(0);
}

//
// TODO: Change the velocity visualization to arrows, instead of the current lines.
// 
// Render velocity vectors
void renderVelocityVectors()
{
    std::vector<float> vecData;
    for (int j = 0; j < Ny; j += 5) {
        for (int i = 0; i < Nx; i += 5) {
            // Find actual cell center for visualization
            // Cell center = (i+0.5)*dx, (j+0.5)*dy
            float xC = (i + 0.5f) * dx;
            float yC = (j + 0.5f) * dy;
            
            // Approx cell-center velocity = average of face velocities
            float uAvg = 0.5f * (u[idxU(i, j)] + u[idxU(i + 1, j)]);
            float vAvg = 0.5f * (v[idxV(i, j)] + v[idxV(i, j + 1)]);
            
            // Normalize and scale for better visualization
            float magnitude = std::sqrt(uAvg*uAvg + vAvg*vAvg);
            float scale = 0.05f;
            
            if (magnitude > 1e-6f) {
                scale *= (0.5f + 0.5f * magnitude); // Scale arrow length with velocity magnitude
            }
            
            float x2 = xC + scale * uAvg;
            float y2 = yC + scale * vAvg;
            
            // Add the two points together, draw as line
            vecData.push_back(xC);
            vecData.push_back(yC);
            vecData.push_back(x2);
            vecData.push_back(y2);
        }
    }

    // Bind a temporary VAO/VBO
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vecData.size() * sizeof(float), vecData.data(), GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Draw as lines
    glLineWidth(3.0f);
    glDrawArrays(GL_LINES, 0, (GLsizei)(vecData.size() / 2));

    // Cleanup
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
}

// Get RGB color from a colormap
static void getColormapRGB(float val, float& r, float& g, float& b)
{
	// Colors from 0 to 1 : Blue -> Cyan -> Green -> Yellow -> Red
    val = std::fmax(0.0f, std::fmin(val, 1.0f));

    float fourVal = 4.0f * val;

    if (val < 0.25f) {
        // 0.00 -> 0.25 : Blue -> Cyan
        r = 0.0f;
        g = fourVal;
        b = 1.0f;
    }
    else if (val < 0.50f) {
        // 0.25 -> 0.50 : Cyan -> Green
        r = 0.0f;
        g = 1.0f;
        b = 1.0f - (fourVal - 1.0f);
    }
    else if (val < 0.75f) {
        // 0.50 -> 0.75 : Green -> Yellow
        r = (fourVal - 2.0f);
        g = 1.0f;
        b = 0.0f;
    }
    else {
        // 0.75 -> 1.00 : Yellow -> Red
        r = 1.0f;
        g = 1.0f - (fourVal - 3.0f);
        b = 0.0f;
    }
}

// Render scalar field (velocity magnitude) using a colormap
void renderScalarField(Shader& scalarShader, const glm::mat4& projection,
                       const glm::mat4& view, const glm::mat4& model)
{

    // If first time, generate VAO/VBO/EBO
    if (!scalarFieldInitialized)
    {
        glGenVertexArrays(1, &scalarVAO);
        glGenBuffers(1, &scalarVBO);
        glGenBuffers(1, &scalarEBO);
        scalarFieldInitialized = true;
    }

    std::vector<float> vertexData;
    vertexData.reserve((Nx + 1) * (Ny + 1) * 5);
    // Store: x, y, r, g, b; 5 floats per vertex

    // Create index buffer to form triangles:
	// For Nx * Ny cells, each cell has 2 triangles, total 2 triangles * Nx * Ny, 2 triangles = 6 verticies, so: 6 * Nx * Ny space is needed
    std::vector<unsigned int> indices;
    indices.reserve(Nx * Ny * 6);

    // Find the "max velocity" for normalizing the color scale
    float maxVel = 1e-6f;
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            // approximate cell-center velocity
            float uAvg = 0.5f * (u[idxU(i, j)] + u[idxU(i + 1, j)]);
            float vAvg = 0.5f * (v[idxV(i, j)] + v[idxV(i, j + 1)]);
            float mag = std::sqrt(uAvg * uAvg + vAvg * vAvg);
            if (mag > maxVel) maxVel = mag;
        }
    }

    // Build the (Nx+1)*(Ny+1) vertices
    // Approximate velocity at corners by averaging the 4 surrounding cell centers.
    // For corners at domain edges, average whatever cells exist.
    for (int j = 0; j <= Ny; j++) {
        for (int i = 0; i <= Nx; i++) {
            float x = i * dx;
            float y = j * dy;

            // Average the neighbors to get velocities
            float sumMag = 0.0f;
            int count = 0;
            // Look at the 4 cell centers around (i,j):
            // Cell indices: (i-1, j-1), (i, j-1), (i-1, j), (i, j)
            // each must be valid: 0 <= i < Nx, 0 <= j < Ny
            for (int dj = -1; dj <= 0; dj++) {
                for (int di = -1; di <= 0; di++) {
                    int ci = i + di;
                    int cj = j + dj;
                    if (ci >= 0 && ci < Nx && cj >= 0 && cj < Ny) {
                        float uAvg = 0.5f * (u[idxU(ci, cj)] + u[idxU(ci + 1, cj)]);
                        float vAvg = 0.5f * (v[idxV(ci, cj)] + v[idxV(ci, cj + 1)]);
                        float mag = std::sqrt(uAvg * uAvg + vAvg * vAvg);
                        sumMag += mag;
                        count++;
                    }
                }
            }
            float cornerMag = (count > 0) ? sumMag / (float)count : 0.0f;
            float normalized = cornerMag / maxVel;

            // Get color from the colormap function
            float R, G, B;
            getColormapRGB(normalized, R, G, B);

            // Push back x, y, r, g, b
            vertexData.push_back(x);
            vertexData.push_back(y);
            vertexData.push_back(R);
            vertexData.push_back(G);
            vertexData.push_back(B);
        }
    }

    // Now build the index buffer (two triangles per cell)
    // For each cell (i,j), the corners are:
    // top-left = j*(Nx+1) + i
    // top-right = j*(Nx+1) + (i+1)
    // bottom-left = (j+1)*(Nx+1) + i
    // bottom-right = (j+1)*(Nx+1) + (i+1)
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            unsigned int topLeft = j * (Nx + 1) + i;
            unsigned int topRight = j * (Nx + 1) + (i + 1);
            unsigned int bottomLeft = (j + 1) * (Nx + 1) + i;
            unsigned int bottomRight = (j + 1) * (Nx + 1) + (i + 1);

            // Triangle 1: topLeft -> bottomLeft -> topRight
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            // Triangle 2: topRight -> bottomLeft -> bottomRight
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    // Bring the data to GPU
    glBindVertexArray(scalarVAO);

    glBindBuffer(GL_ARRAY_BUFFER, scalarVBO);
    glBufferData(GL_ARRAY_BUFFER,
        vertexData.size() * sizeof(float),
        vertexData.data(),
        GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scalarEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
        indices.size() * sizeof(unsigned int),
        indices.data(),
        GL_DYNAMIC_DRAW);

    // Layout: position is 2 floats, color is 3 floats so total of 5 floats per vertex
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    glBindVertexArray(scalarVAO);
    glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

glm::mat4 windowCentered(GLFWwindow* window) {
    // Get current framebuffer size
    static int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    float aspect = (float)width / (float)height;

    // Define simulation boundaries
    float simMinX = -0.1f, simMaxX = 1.1f;
    float simMinY = -0.1f, simMaxY = 1.1f;
    float centerX = (simMinX + simMaxX) * 0.5f;
    float centerY = (simMinY + simMaxY) * 0.5f;
    float halfWidth = (simMaxX - simMinX) * 0.5f;
    float halfHeight = (simMaxY - simMinY) * 0.5f;

    float newHalfWidth, newHalfHeight;
    if (aspect >= 1.0f) {
        // Window is wider than tall: extend horizontally.
        newHalfWidth = halfWidth * aspect;
        newHalfHeight = halfHeight;
    }
    else {
        // Window is taller than wide: extend vertically.
        newHalfWidth = halfWidth;
        newHalfHeight = halfHeight / aspect;
    }

    // Create a new orthographic projection matrix that stays centered.
    return glm::ortho(
        centerX - newHalfWidth, centerX + newHalfWidth,
        centerY - newHalfHeight, centerY + newHalfHeight,
        -1.0f, 1.0f);
}
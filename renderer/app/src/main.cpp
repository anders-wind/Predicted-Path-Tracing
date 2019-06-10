#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>


void main_loop(GLFWwindow* window)
{
    // a triangle
    constexpr int element_size = 2;
    constexpr int number_of_elements = 3;
    float positions[element_size * number_of_elements] = {
        -0.5f, -0.5f, 0.0f, 0.5f, 0.5f, -0.5f,
    };

    unsigned int buffer; // this is the handle for the buffer
    glGenBuffers(1, &buffer); // create the buffer and write the id
    glBindBuffer(GL_ARRAY_BUFFER, buffer); // Select the buffer as active.
    glBufferData(GL_ARRAY_BUFFER, element_size * number_of_elements * sizeof(float), positions, GL_STATIC_DRAW); // static for update once, dynamic for updated many time
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, element_size, GL_FLOAT, GL_FALSE, element_size * sizeof(float), (const void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0); // bind no buffer.

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        glBindBuffer(GL_ARRAY_BUFFER, buffer); // Select the buffer as active.
        glDrawArrays(GL_TRIANGLES, 0, number_of_elements);
        glBindBuffer(GL_ARRAY_BUFFER, 0); // bind no buffer.

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }
}

GLFWwindow* init_window()
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        exit(-1);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "PPT", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(-1);
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    return window;
}

void glew_init()
{
    // Init Glew
    if (glewInit() != GLEW_OK || !glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        exit(0);
    }
}

void shutdown()
{
    glfwTerminate();
}

int main(void)
{
    GLFWwindow* window = init_window();

    glew_init();

    main_loop(window);

    shutdown();

    return 0;
}
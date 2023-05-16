#include <assert.h>
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "params.h"
#include "photon.h"

#define PHOTON_CAP 1 << 16
#define MAX_PHOTONS_PER_FRAME 20

static float heats[SHELLS];
static float _heats_squared[SHELLS];
static int remaining_photons = PHOTON_CAP;

// clang-format off
// covers the entire screen with 2 triangles
static const char *VSHADER = ""
"#version 430\n"

"vec2 vertices[4] = {\n"
"    {-1.0,  1.0},\n"
"    {-1.0, -1.0},\n"
"    { 1.0, -1.0},\n"
"    { 1.0,  1.0}\n"
"};\n"

"uint indices[6] = {0, 1, 2, 0, 2, 3};\n"

"void main() {\n"
"    gl_Position = vec4(vertices[indices[gl_VertexID]], 0.0, 1.0);\n"
"}"
;

// maps a pixel to an array index based on it's distance from the origin and
// the heat value of that index to a shade of red
static const char *FSHADER = ""
"#version 430\n"

// magic constant - max distance from origin = length(+/-vec2(0.5))
"#define MC 0.7071067811865476f\n"

"out vec4 frag_color;\n"

"layout(std430, binding = 0) readonly buffer ssbo {\n"
"    float heats[];\n"
"} shells;\n"

"void main() {\n"
"    vec2 uv = gl_FragCoord.xy / vec2(800);\n"

"    float dr = length(uv - vec2(0.5));\n"

"    int heat_id = int((dr / MC) * float(shells.heats.length() - 1));\n"

"    float heat = shells.heats[heat_id];\n"

    // logistic functions to put less weight in the difference between huge colors
    // try plotting it in a graph calculator, see what happens with different k's
"    float L = 2.0;\n"
"    float b = 1.0;\n"
"    float k = 0.004;\n"
"    float heat_fit = L / (1.0 + b * exp(-k * heat)) - 1.0;\n"

"    frag_color = vec4(heat_fit, 0.0, 0.0, 1.0);\n"
"}"
;
// clang-format on

void update(void)
{
    if (remaining_photons <= 0) {
        return;
    }

    int remaining_photons_in_frame = MAX_PHOTONS_PER_FRAME;

    while (remaining_photons > 0 && remaining_photons_in_frame > 0) {
        --remaining_photons;
        --remaining_photons_in_frame;

        photon(heats, _heats_squared);
    }

    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(heats), heats);
}

int main(void)
{
    glfwInit();

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(800, 800, "tiny mc", NULL, NULL);
    assert(window != NULL);

    glfwMakeContextCurrent(window);
    glewInit();

    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &VSHADER, NULL);
    glCompileShader(vertex_shader);

    GLint status;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &status);
    assert(status);

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &FSHADER, NULL);
    glCompileShader(fragment_shader);

    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &status);
    assert(status);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    assert(status);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    glUseProgram(program);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glViewport(0, 0, 800, 800);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    glfwShowWindow(window);

    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(heats), heats, GL_DYNAMIC_DRAW);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        update();

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
    }

    glDeleteBuffers(1, &ssbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
}

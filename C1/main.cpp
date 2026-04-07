#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>
#include <cmath>
#include <numeric>

#include "Shader.h"

using namespace std;

//Config
constexpr int NUM_NODOS = 12;
constexpr int NUM_COLORES = 3;
float zoom = 1.0f;
int nodoInicialH2 = -1;
int nodoBajoMouse = -1;
int conteoBacktrack = 0;




//Colores---------------
float coloresRGB[5][3] = {
    {1.0f, 0.2f, 0.2f},//rojo
    {0.2f, 1.0f, 0.2f},//verde
    {0.2f, 0.4f, 1.0f},//azul
    {1.0f, 1.0f, 0.1f},//amarillo
    {0.1f, 1.0f, 1.0f} //cyan
};
float coloresRGBOscuro[5][3] = {
    {0.5f, 0.05f, 0.05f}, {0.05f, 0.5f, 0.05f}, {0.05f, 0.15f, 0.5f},
    {0.5f, 0.5f, 0.0f},   {0.0f, 0.4f, 0.4f}
};



struct Nodo {
    int id;
    float x, y;
    int color;
    vector<int> vecinos;

    Nodo(int id, float x, float y)
        : id(id), x(x), y(y), color(-1) {}

    bool estaConectadoCon(int otherId) const {
        return find(vecinos.begin(), vecinos.end(), otherId) != vecinos.end();
    }
};

vector<Nodo> nodos;
GLuint nodeVAO, nodeVBO, edgeVAO, edgeVBO;
//Config grafico
bool grafoGenerado = false, coloreoAplicado = false, solucionEncontrada = false;
enum Heuristica { HEURISTICA_1, HEURISTICA_2 };

//aristas Opengl
void actualizarBufferAristas() {
    vector<float> edgeVerts;
    for (const auto& n : nodos) {
        for (int vid : n.vecinos) {
            if (vid > n.id) {
                edgeVerts.push_back(n.x); edgeVerts.push_back(n.y);
                edgeVerts.push_back(nodos[vid].x); edgeVerts.push_back(nodos[vid].y);
            }
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, edgeVBO);
    glBufferData(GL_ARRAY_BUFFER, edgeVerts.size() * sizeof(float), edgeVerts.data(), GL_DYNAMIC_DRAW);
}

//Posicion x y aleatorio
void generarPosicionesTablero() {
    mt19937 gen(static_cast<unsigned>(chrono::high_resolution_clock::now().time_since_epoch().count()));
    const int COLS = 200;//colsxfilas
    const int FILAS = 200;
    float cellW = 1.7f / COLS;//tamaño celdas
    float cellH = 1.7f / FILAS;

    vector<pair<int,int>> celdas;
    for (int r = 0; r < FILAS; r++)
        for (int c = 0; c < COLS; c++)
            celdas.push_back({r, c});
    shuffle(celdas.begin(), celdas.end(), gen);

    uniform_real_distribution<float> jitterX(0.15f, 0.85f);
    uniform_real_distribution<float> jitterY(0.15f, 0.85f);

    for (int i = 0; i < NUM_NODOS; i++) {
        auto [fila, col] = celdas[i];
        nodos[i].x = -0.85f + (col + jitterX(gen)) * cellW;
        nodos[i].y = -0.85f + (fila + jitterY(gen)) * cellH;
    }
}

//dibujo grafo mas ordenado
void aplicarForceDirected(int iteraciones = 300) {
    float k            = 0.36f;
    float temperatura  = 0.15f;
    float enfriamiento = 0.98f;

    for (int iter = 0; iter < iteraciones; iter++) {
        vector<float> fx(NUM_NODOS, 0.0f);
        vector<float> fy(NUM_NODOS, 0.0f);

        // Repulsión entre todos los pares
        for (int i = 0; i < NUM_NODOS; i++) {
            for (int j = i + 1; j < NUM_NODOS; j++) {
                float dx = nodos[i].x - nodos[j].x;
                float dy = nodos[i].y - nodos[j].y;
                float dist = sqrt(dx*dx + dy*dy);
                if (dist < 0.001f) dist = 0.001f;

                float fuerza = (k * k) / dist;
                float nx = (dx / dist) * fuerza;
                float ny = (dy / dist) * fuerza;

                fx[i] += nx;  fy[i] += ny;
                fx[j] -= nx;  fy[j] -= ny;
            }
        }

        // Atracción por aristas
        for (int i = 0; i < NUM_NODOS; i++) {
            for (int j : nodos[i].vecinos) {
                if (j <= i) continue;
                float dx = nodos[j].x - nodos[i].x;
                float dy = nodos[j].y - nodos[i].y;
                float dist = sqrt(dx*dx + dy*dy);
                if (dist < 0.001f) dist = 0.001f;

                float fuerza = (dist * dist) / k;
                float nx = (dx / dist) * fuerza;
                float ny = (dy / dist) * fuerza;

                fx[i] += nx;  fy[i] += ny;
                fx[j] -= nx;  fy[j] -= ny;
            }
        }

        // Aplicar desplazamientos con límite de temperatura
        for (int i = 0; i < NUM_NODOS; i++) {
            float mag = sqrt(fx[i]*fx[i] + fy[i]*fy[i]);
            if (mag < 0.001f) continue;

            float escala = min(mag, temperatura) / mag;
            nodos[i].x += fx[i] * escala;
            nodos[i].y += fy[i] * escala;

            nodos[i].x = max(-0.9f, min(0.9f, nodos[i].x));
            nodos[i].y = max(-0.9f, min(0.9f, nodos[i].y));
        }

        temperatura *= enfriamiento;
    }

    actualizarBufferAristas();
}

void generarGrafo() {
    nodos.clear();
    mt19937 gen(static_cast<unsigned>(chrono::high_resolution_clock::now().time_since_epoch().count()));

    for (int i = 0; i < NUM_NODOS; i++)
        nodos.emplace_back(i, 0.0f, 0.0f);

    generarPosicionesTablero();

    // Elegir densidad aleatoria
    uniform_int_distribution<int> nivelDist(0, 2);
    int nivel = nivelDist(gen);

    float probabilidadConexion;
    switch (nivel) {
        case 0: probabilidadConexion = 0.20f; break;
        case 1: probabilidadConexion = 0.35f; break;
        case 2: probabilidadConexion = 0.50f; break;
    }

    uniform_real_distribution<float> probDist(0.0f, 1.0f);

    for (int i = 0; i < NUM_NODOS; i++) {
        for (int j = i + 1; j < NUM_NODOS; j++) {
            if ((int)nodos[i].vecinos.size() >= 5) break;
            if ((int)nodos[j].vecinos.size() >= 5) continue;
            if (probDist(gen) < probabilidadConexion) {
                nodos[i].vecinos.push_back(j);
                nodos[j].vecinos.push_back(i);
            }
        }
    }

    auto obtenerComponente = [&](int inicio) {
        vector<bool> vis(NUM_NODOS, false);
        vector<int> componente;
        vector<int> cola = {inicio};
        vis[inicio] = true;
        while (!cola.empty()) {
            int u = cola.back(); cola.pop_back();
            componente.push_back(u);
            for (int v : nodos[u].vecinos)
                if (!vis[v]) { vis[v] = true; cola.push_back(v); }
        }
        return componente;
    };

    auto obtenerTodasComponentes = [&]() {
        vector<bool> vis(NUM_NODOS, false);
        vector<vector<int>> componentes;
        for (int i = 0; i < NUM_NODOS; i++) {
            if (!vis[i]) {
                vector<int> comp = obtenerComponente(i);
                for (int u : comp) vis[u] = true;
                componentes.push_back(comp);
            }
        }
        return componentes;
    };

    while (true) {
        vector<vector<int>> componentes = obtenerTodasComponentes();
        if (componentes.size() == 1) break;

        shuffle(componentes.begin(), componentes.end(), gen);

        bool conectado = false;
        for (int ci = 0; ci < (int)componentes.size() - 1 && !conectado; ci++) {
            vector<int> compA = componentes[ci];
            vector<int> compB = componentes[ci + 1];

            shuffle(compA.begin(), compA.end(), gen);
            shuffle(compB.begin(), compB.end(), gen);

            for (int a : compA) {
                if ((int)nodos[a].vecinos.size() >= 5) continue;
                for (int b : compB) {
                    if ((int)nodos[b].vecinos.size() >= 5) continue;
                    if (nodos[a].estaConectadoCon(b)) continue;
                    nodos[a].vecinos.push_back(b);
                    nodos[b].vecinos.push_back(a);
                    conectado = true;
                    break;
                }
                if (conectado) break;
            }

            if (!conectado) {
                for (int a : compA) {
                    for (int b : compB) {
                        if (nodos[a].estaConectadoCon(b)) continue;
                        nodos[a].vecinos.push_back(b);
                        nodos[b].vecinos.push_back(a);
                        conectado = true;
                        break;
                    }
                    if (conectado) break;
                }
            }
        }
    }

    for (int i = 0; i < NUM_NODOS; i++) {
        if ((int)nodos[i].vecinos.size() >= 2) continue;

        vector<pair<float, int>> candidatos;
        for (int j = 0; j < NUM_NODOS; j++) {
            if (j == i) continue;
            if (nodos[i].estaConectadoCon(j)) continue;
            if ((int)nodos[j].vecinos.size() >= 5) continue;
            float dx = nodos[i].x - nodos[j].x;
            float dy = nodos[i].y - nodos[j].y;
            candidatos.push_back({dx*dx + dy*dy, j});
        }
        sort(candidatos.begin(), candidatos.end());

        for (auto& [dist, j] : candidatos) {
            if ((int)nodos[i].vecinos.size() >= 2) break;
            if ((int)nodos[i].vecinos.size() >= 5) break;
            nodos[i].vecinos.push_back(j);
            nodos[j].vecinos.push_back(i);
        }
    }

    grafoGenerado = true;
    coloreoAplicado = false;
    nodoInicialH2 = -1;

    aplicarForceDirected(300);

    int totalAristas = 0;
    int gradoMin = NUM_NODOS, gradoMax = 0;
    for (auto& n : nodos) {
        totalAristas += n.vecinos.size();
        gradoMin = min(gradoMin, (int)n.vecinos.size());
        gradoMax = max(gradoMax, (int)n.vecinos.size());
    }
    totalAristas /= 2;

    cout << "\n[GRAFO GENERADO]" << endl;
    cout << "  Nodos               : " << NUM_NODOS << endl;
    cout << "  Aristas             : " << totalAristas << endl;
    cout << "  Grado minimo        : " << gradoMin << endl;
    cout << "  Grado maximo        : " << gradoMax << endl;
    cout << "  Grados (nodo:grado) : ";
    for (auto& n : nodos) cout << n.id << ":" << n.vecinos.size() << " ";
    cout << endl;
}

//backtrack
bool ColorV(int nodoId, int color) {
    for (int v : nodos[nodoId].vecinos)
        if (nodos[v].color == color) return false;
    return true;
}

bool backtrack(const vector<int>& orden, int idx) {
    conteoBacktrack++;
    if (idx == (int)orden.size()) return true;
    int nId = orden[idx];
    for (int c = 0; c < NUM_COLORES; c++) {
        if (ColorV(nId, c)) {
            nodos[nId].color = c;
            if (backtrack(orden, idx + 1)) return true;
            nodos[nId].color = -1;
        }
    }
    return false;
}

//Heuristica
void aplicarColoreo(Heuristica h) {
    for (auto& n : nodos) n.color = -1;
    conteoBacktrack = 0;
    vector<int> orden;

    if (h == HEURISTICA_1) {
        cout << "\n[HEURISTICA 1 - Mayor Grado Primero]" << endl;
        orden.resize(NUM_NODOS);
        iota(orden.begin(), orden.end(), 0);
        sort(orden.begin(), orden.end(), [](int a, int b) {
            if (nodos[a].vecinos.size() != nodos[b].vecinos.size())
                return nodos[a].vecinos.size() > nodos[b].vecinos.size();
            return a < b; // desempate por id
        });
        cout << "Orden de asignacion (id:grado): ";
        for (int i : orden) cout << i << ":" << nodos[i].vecinos.size() << " ";
        cout << endl;

    } else {
        cout << "\n[HEURISTICA 2 - BFS desde nodo aleatorio]" << endl;
        mt19937 gen(static_cast<unsigned>(chrono::high_resolution_clock::now().time_since_epoch().count()));
        int inicio = uniform_int_distribution<int>(0, NUM_NODOS - 1)(gen);
        nodoInicialH2 = inicio;
        cout << "Nodo inicial BFS: " << inicio << endl;

        vector<bool> vis(NUM_NODOS, false);
        vector<int> q = {inicio};
        vis[inicio] = true;
        while (!q.empty()) {
            int u = q.front(); q.erase(q.begin()); orden.push_back(u);
            for (int v : nodos[u].vecinos)
                if (!vis[v]) { vis[v] = true; q.push_back(v); }
        }
        for (int i = 0; i < NUM_NODOS; i++) if (!vis[i]) orden.push_back(i);

        cout << "Orden BFS: ";
        for (int i : orden) cout << i << " ";
        cout << endl;
    }

    auto t0 = chrono::high_resolution_clock::now();
    solucionEncontrada = backtrack(orden, 0);
    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();

    coloreoAplicado = true;
    cout << "--- Resultado Backtracking ---" << endl;
    cout << "Solucion encontrada: " << (solucionEncontrada ? "SI" : "NO") << endl;
    cout << "Llamadas al backtrack: " << conteoBacktrack << endl;
    cout << "Tiempo: " << ms << " ms" << endl;
    if (solucionEncontrada) {
        cout << "Colores asignados (nodo:color): ";
        for (auto& n : nodos) cout << n.id << ":" << n.color << " ";
        cout << endl;
    }
    cout << "------------------------------" << endl;
}

//mouse
void manejarMouse(GLFWwindow* window) {
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    int w, h;
    glfwGetWindowSize(window, &w, &h);

    float mouseNDC_X = (float)((mx / w) * 2.0f - 1.0f);
    float mouseNDC_Y = (float)(1.0f - (my / h) * 2.0f);
    float aspect     = (float)w / (float)h;
    float worldX     = mouseNDC_X / (zoom * aspect);
    float worldY     = mouseNDC_Y / zoom;

    nodoBajoMouse = -1;
    float radioDeteccion = 0.05f / zoom;

    for (const auto& n : nodos) {
        float dx = n.x - worldX, dy = n.y - worldY;
        if ((dx*dx + dy*dy) < (radioDeteccion * radioDeteccion)) {
            nodoBajoMouse = n.id;
            break;
        }
    }

    string title = "CSP Grafo";
    if (nodoBajoMouse != -1) {
        title += " | NODO: " + to_string(nodoBajoMouse);
        title += " | Color: ";
        title += (coloreoAplicado && nodos[nodoBajoMouse].color != -1)
                 ? to_string(nodos[nodoBajoMouse].color) : "N/A";
        title += " | Vecinos(" + to_string(nodos[nodoBajoMouse].vecinos.size()) + "): ";
        for (int v : nodos[nodoBajoMouse].vecinos) title += to_string(v) + " ";
    }
    glfwSetWindowTitle(window, title.c_str());
}

//teclado
void manejarTeclado(GLFWwindow* window) {
    static bool pG = false, p1 = false, p2 = false, pR = false;

    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        if (!pG) { generarGrafo(); pG = true; }
    } else pG = false;

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        if (!p1 && grafoGenerado) { aplicarColoreo(HEURISTICA_1); p1 = true; }
    } else p1 = false;

    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        if (!p2 && grafoGenerado) { aplicarColoreo(HEURISTICA_2); p2 = true; }
    } else p2 = false;

    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        if (!pR) {
            for (auto& n : nodos) n.color = -1;
            coloreoAplicado = false;
            nodoInicialH2 = -1;
            cout << "\n[RESET] Colores eliminados.\n";
            pR = true;
        }
    } else pR = false;
}

//Grafica
void dibujar(Shader& shader, int w, int h) {
    glViewport(0, 0, w, h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    if (!grafoGenerado) return;

    shader.use();
    shader.setFloat("zoom", zoom);
    shader.setFloat("aspect", (float)w / h);

    // Aristas
    shader.setVec3("color", 0.4f, 0.4f, 0.4f);
    glBindVertexArray(edgeVAO);
    int eCount = 0;
    for (auto& n : nodos) for (int v : n.vecinos) if (v > n.id) eCount++;
    glDrawArrays(GL_LINES, 0, eCount * 2);

    // Nodos
    for (const auto& n : nodos) {
        float r = 0.7f, g = 0.7f, b = 0.7f, tam = 16.0f;

        if (coloreoAplicado && solucionEncontrada && n.color != -1) {
            r = coloresRGB[n.color][0];
            g = coloresRGB[n.color][1];
            b = coloresRGB[n.color][2];
        }

        if (n.id == nodoInicialH2) {
            tam = 30.0f;
            if (coloreoAplicado && solucionEncontrada && n.color != -1) {
                r = coloresRGBOscuro[n.color][0];
                g = coloresRGBOscuro[n.color][1];
                b = coloresRGBOscuro[n.color][2];
            } else {
                r = 0.35f; g = 0.35f; b = 0.35f;
            }
        } else if (n.id == nodoBajoMouse) {
            tam = 28.0f;
            r = min(r + 0.3f, 1.0f);
            g = min(g + 0.3f, 1.0f);
            b = min(b + 0.3f, 1.0f);
        }

        glPointSize(tam);
        shader.setVec3("color", r, g, b);
        float p[] = { n.x, n.y };
        glBindBuffer(GL_ARRAY_BUFFER, nodeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(p), p, GL_DYNAMIC_DRAW);
        glBindVertexArray(nodeVAO);
        glDrawArrays(GL_POINTS, 0, 1);
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    zoom += (float)yoffset * 0.1f;
    if (zoom < 0.1f) zoom = 0.1f;
}

void inicializarBuffers() {
    glGenVertexArrays(1, &nodeVAO); glGenBuffers(1, &nodeVBO);
    glBindVertexArray(nodeVAO); glBindBuffer(GL_ARRAY_BUFFER, nodeVBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glGenVertexArrays(1, &edgeVAO); glGenBuffers(1, &edgeVBO);
    glBindVertexArray(edgeVAO); glBindBuffer(GL_ARRAY_BUFFER, edgeVBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "CSP Graph Coloring", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetScrollCallback(window, scroll_callback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    glEnable(GL_PROGRAM_POINT_SIZE);
    Shader shader("vertex.glsl", "fragment.glsl");
    inicializarBuffers();

    cout << "========================================" << endl;
    cout << "  CSP - Coloreo de Grafos con Backtrack " << endl;
    cout << "========================================" << endl;
    cout << " [G] Generar nuevo grafo"                    << endl;
    cout << " [1] Heuristica 1: Mayor grado primero"      << endl;
    cout << " [2] Heuristica 2: BFS desde nodo aleatorio" << endl;
    cout << " [R] Reset colores"                          << endl;
    cout << " [Scroll] Zoom"                              << endl;
    cout << "========================================" << endl;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        manejarTeclado(window);
        manejarMouse(window);

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        dibujar(shader, w, h);
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}

#include <iostream>
#include <fstream>
#include <getopt.h>

#include "main.hpp"
#include "kernels.hpp"
#include "bitmap_image.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    // print help
    if (argc == 1) {
        printParameters();
        return 0;
    }

    // get program options
    int width = 1920;
    int height = 1080;

    int iterations = 200;

    float c_x = 0.285;
    float c_y = 0.01;

    float top = 1.2;
    float bottom = -1.2;
    float left = -2.1;
    float right = 2.1;

    char opt;
    while ((opt = getopt(argc, argv, "w:h:x:y:i:t:b:l:r:")) != -1) {
        switch (opt) {
            case 'w': {
                width = (int) strtol(optarg, nullptr, 10);
                break;
            }
            case 'h': {
                height = (int) strtol(optarg, nullptr, 10);
                break;
            }
            case 'x': {
                c_x = strtof(optarg, nullptr);
                break;
            }
            case 'y': {
                c_y = strtof(optarg, nullptr);
                break;
            }
            case 'i': {
                iterations = strtol(optarg, nullptr, 10);
                break;
            }
            case 't': {
                top = strtof(optarg, nullptr);
                break;
            }
            case 'b': {
                bottom = strtof(optarg, nullptr);
                break;
            }
            case 'l': {
                left = strtof(optarg, nullptr);
                break;
            }
            case 'r': {
                right = strtof(optarg, nullptr);
                break;
            }
            default: {
                break;
            }
        }
    }

    // get color map
    vector<unsigned char> h_color_map;

    ifstream f;
    f.open("color_maps/custom.txt");
    int c;
    while (f >> c) {
        h_color_map.push_back(c);
    }

    char *h_colors = (char *) malloc(width * height * 3 * sizeof(char));

    // allocate device memory
    float *d_coords;
    int *d_steps;
    unsigned char *d_colors;
    unsigned char *d_color_map;

    cudaMalloc(&d_coords, width * height * 2 * sizeof(float));
    cudaMalloc(&d_steps, width * height * sizeof(int));
    cudaMalloc(&d_colors, width * height * 3 * sizeof(unsigned char));
    cudaMalloc(&d_color_map, 256 * 3 * sizeof(unsigned char));

    // initialize device memory
    dim3 grid((width + 32 - 1) / 32, (height + 32 - 1) / 32);
    dim3 block(32, 32);

    cudaMemset(d_steps, -1, width * height * sizeof(int));
    initializeCoords<<<grid, block>>>(d_coords,
                                      width,
                                      height,
                                      top,
                                      bottom,
                                      left,
                                      right);

    // run main loop
    for (int i = 0; i < iterations; i++) {
        iterateCoordinates<<<grid, block>>>(d_coords,
                                            d_steps,
                                            width,
                                            height,
                                            c_x,
                                            c_y,
                                            i);
    }

    // apply color
    cudaMemcpy(d_color_map, h_color_map.data(), 256 * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    setColors<<<grid, block>>>(d_steps,
                               d_colors,
                               d_color_map,
                               width,
                               height);

    // bring back data
    cudaMemcpy(h_colors, d_colors, width * height * 3 * sizeof(char), cudaMemcpyDeviceToHost);

    // create image
    bitmap_image image(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int idx = (y * width + x) * 3;
            image.set_pixel(x, y, h_colors[idx], h_colors[idx + 1], h_colors[idx + 2]);
        }
    }

    image.save_image("output.bmp");

    return 0;
}

void printParameters() {
    cout << "-w W   Width in pixels" << endl;
    cout << "-h H   Height in pixels." << endl;
    cout << "-x X   Real part of seed." << endl;
    cout << "-y Y   Imaginary part of seed." << endl;
    cout << "-i I   Number of iterations." << endl;
    cout << "-t T   Y-coordinate of top." << endl;
    cout << "-b B   Y-coordinate of bottom." << endl;
    cout << "-l L   X-coordinate of left." << endl;
    cout << "-r R   X-coordinate of right." << endl;
}
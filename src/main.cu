#include <iostream>
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
    long width = 1280;
    long height = 720;

    long iterations = 100;

    float c_x = 0;
    float c_y = 0;

    float top = 1;
    float bottom = -1;
    float left = -1;
    float right = 1;

    char opt;
    while ((opt = getopt(argc, argv, "w:h:x:y:i:t:b:l:r:")) != -1) {
        switch (opt) {
            case 'w': {
                width = strtol(optarg, nullptr, 10);
                break;
            }
            case 'h': {
                height = strtol(optarg, nullptr, 10);
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

    char *h_colors = (char *) malloc(width * height * 3 * sizeof(char));

    // allocate device memory
    float *d_coords;
    int *d_steps;
    char *d_colors;

    cudaMalloc(&d_coords, width * height * 2 * sizeof(float));
    cudaMalloc(&d_steps, width * height * sizeof(int));
    cudaMalloc(&d_colors, width * height * 3 * sizeof(char));

    // initialize device memory
    dim3 grid((width + 32 - 1) / 32, (height + 32 - 1) / 32);
    dim3 block(32, 32);

    cudaMemset(d_steps, 0, width * height * sizeof(int));
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
    setColors<<<grid, block>>>(d_steps,
                               d_colors,
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
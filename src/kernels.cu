#include "kernels.hpp"

__global__ void
iterateCoordinates(float *d_coords,
                   int *d_steps,
                   const long width,
                   const long height,
                   const float c_x,
                   const float c_y,
                   const long iteration) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        unsigned int ids = idy * width + idx;
        unsigned int idc = ids * 2;
        float x = d_coords[idc];
        float y = d_coords[idc + 1];

        // check if escaped
        if (d_steps[ids] == 0) {
            if (x * x + y * y > 4) {
                d_steps[ids] = iteration;
            } else {
                d_coords[idc] = x * x - y * y + c_x;
                d_coords[idc + 1] = 2 * x * y + c_y;
            }
        }
    }
}

__global__ void
initializeCoords(float *d_coords,
                 const long width,
                 const long height,
                 const float top,
                 const float bottom,
                 const float left,
                 const float right) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        unsigned int idc = (idy * width + idx) * 2;
        d_coords[idc] = left + (right - left) / (float) width * idx;
        d_coords[idc + 1] = top - (top - bottom) / (float) height * idy;
    }
}

__global__ void
setColors(const int *d_steps,
          char *d_colors,
          const long width,
          const long height) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        unsigned int ids = idy * width + idx;
        unsigned int idc = ids * 3;
        d_colors[idc] = d_steps[ids];
        d_colors[idc + 1] = d_steps[ids];
        d_colors[idc + 2] = d_steps[ids];
    }
}

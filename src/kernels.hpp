__global__ void
iterateCoordinates(const float *d_coords,
                   int *d_steps,
                   int width,
                   int height,
                   float c_x,
                   float c_y,
                   int iterations);

__global__ void
initializeCoords(float *d_coords,
                 int width,
                 int height,
                 float top,
                 float bottom,
                 float left,
                 float right);

__global__ void
setColors(const int *d_steps,
          unsigned char *d_colors,
          const unsigned char *d_color_map,
          int width,
          int height);

__global__ void
combinedKernel(const float *d_coords,
               int *d_steps,
               const int width,
               const int height,
               const float c_x,
               const float c_y,
               const int iterations,
               unsigned char *d_colors,
               const unsigned char *d_color_map,
               const float top,
               const float bottom,
               const float left,
               const float right);
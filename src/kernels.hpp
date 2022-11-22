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

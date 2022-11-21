__global__ void
iterateCoordinates(float *d_coords,
                   int *d_steps,
                   const long width,
                   const long height,
                   const float c_x,
                   const float c_y,
                   const long iteration);

__global__ void
initializeCoords(float *d_coords,
                 const long width,
                 const long height,
                 const float top,
                 const float bottom,
                 const float left,
                 const float right);

__global__ void
setColors(const int *d_steps,
          char *d_colors,
          const long width,
          const long height);

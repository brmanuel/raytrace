#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "include/gpu_trace.hu"

/*
  Error handling helpers
 */
#define gpu_err_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }


inline void
gpu_assert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

__host__ __device__ uint32_t
get_idx_of_first_intersection(sphere *spheres,
                              uint32_t num_spheres,
                              vector ray_start,
                              vector ray_dir,
                              vector *intersection)
{
    float ray_len = sqrt(ray_dir.x * ray_dir.x +
                         ray_dir.y * ray_dir.y +
                         ray_dir.z * ray_dir.z);
    float ray_x_norm = ray_dir.x / ray_len;
    float ray_y_norm = ray_dir.y / ray_len;
    float ray_z_norm = ray_dir.z / ray_len;
    float min_dist = FLT_MAX;
    uint32_t winner_idx = num_spheres;
            
    for (uint32_t sphere_idx = 0; sphere_idx < num_spheres; sphere_idx++){
        sphere sphere = spheres[sphere_idx];
        vector vec_to_center = {
            ray_start.x - sphere.center.x,
            ray_start.y - sphere.center.y,
            ray_start.z - sphere.center.z,
        };
        float a = 1;
        float b = 2*(ray_x_norm * vec_to_center.x +
                     ray_y_norm * vec_to_center.y +
                     ray_z_norm * vec_to_center.z);
        float c = (vec_to_center.x * vec_to_center.x +
                   vec_to_center.y * vec_to_center.y +
                   vec_to_center.z * vec_to_center.z -
                   sphere.radius * sphere.radius);
        float discriminant = b*b - 4*a*c;
        if (discriminant < 0) {
            continue;
        }
        float x1 = (-b - sqrt(discriminant))/(2*a);
        float x2 = (-b + sqrt(discriminant))/(2*a);
        float *x = NULL;
        if (x1 > 0.1 && x1 < min_dist){
            x = &x1;
        }
        else if (x2 > 0.1 && x2 < min_dist){ // x2 must be > x1!
            x = &x2;
        }

        if (x != NULL) {
            min_dist = *x;
            winner_idx = sphere_idx;
            *intersection = (vector){
                ray_start.x + *x * ray_x_norm,
                ray_start.y + *x * ray_y_norm,
                ray_start.z + *x * ray_z_norm,
            };
        }
    }
    return winner_idx;
}
                       

__global__ void
cuda_trace_kernel(sphere *spheres,
                  uint32_t num_spheres,
                  float canvas_min_x,
                  float canvas_max_x,
                  float canvas_min_z,
                  float canvas_max_z,
                  float canvas_y,
                  uint32_t num_pixels_x,
                  uint32_t num_pixels_z,
                  uint32_t *canvas)
{
    // asm: eye is at (0,0,0)
    // asm: canvas is aligned to y-axis
    // asm: no object means black

    uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t grid_size = gridDim.x * blockDim.x;

    while (thread_index < num_pixels_x * num_pixels_z) {

        // initialize canvas to black
        canvas[thread_index] = 67108863;

        vector zero_vec = {0.0,0.0,0.0};
        float canvas_width = canvas_max_x - canvas_min_x;
        float canvas_x_incr = canvas_width / num_pixels_x;
        float canvas_height = canvas_max_z - canvas_min_z;
        float canvas_z_incr = canvas_height / num_pixels_z;
        uint32_t x_idx = thread_index % num_pixels_x;
        uint32_t z_idx = thread_index / num_pixels_x;
    
        vector ray_dir = {
            canvas_min_x + (x_idx+0.5) * canvas_x_incr,
            canvas_y,
            canvas_min_z + (z_idx+0.5) * canvas_z_incr
        };
        vector intersection;
        uint32_t winner_sphere_idx =
            get_idx_of_first_intersection(spheres,
                                          num_spheres,
                                          zero_vec,
                                          ray_dir,
                                          &intersection);

        if (winner_sphere_idx < num_spheres) {
            // intersecting sphere found
            sphere winner_sphere = spheres[winner_sphere_idx];

            // only intersection color
            canvas[thread_index] = winner_sphere.color;
        }

        thread_index += grid_size;
    }
}






/*

  ( 8) Multiprocessors, (192) CUDA Cores/MP:     1536 CUDA Cores
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)

 */
void gpu_trace(sphere **spheres,
               uint32_t num_spheres,
               float canvas_min_x,
               float canvas_max_x,
               float canvas_min_z,
               float canvas_max_z,
               float canvas_y,
               uint32_t num_pixels_x,
               uint32_t num_pixels_z,
               uint32_t *canvas)
{

    // Allocate GPU memory for the spheres and copy them over
    // TODO: use constant memory for the spheres
    sphere *gpu_spheres;
    gpu_err_check(cudaMalloc((void **) &gpu_spheres, sizeof(sphere) * num_spheres));
    for (int i = 0; i < num_spheres; i++){
        gpu_err_check(cudaMemcpy(&gpu_spheres[i],
                                 spheres[i],
                                 sizeof(sphere),
                                 cudaMemcpyHostToDevice));
    }
    
    // Allocate GPU memory for the canvas
    uint32_t *gpu_canvas;
    gpu_err_check(cudaMalloc((void **) &gpu_canvas,
                             sizeof(uint32_t) * num_pixels_x * num_pixels_z));

    
    // Call the kernel using one thread per pixel
    // want multiple warps per block and block size should be
    // a multiple of 32.

    // simply choose 32 threads_per_block initially
    // TODO: try out different threads_per_block
    uint32_t num_threads = num_pixels_x * num_pixels_z;
    uint32_t num_blocks = (num_threads + 31) / 32;
    cuda_trace_kernel<<<num_blocks, 32>>>(gpu_spheres,
                                          num_spheres,
                                          canvas_min_x,
                                          canvas_max_x,
                                          canvas_min_z,
                                          canvas_max_z,
                                          canvas_y,
                                          num_pixels_x,
                                          num_pixels_z,
                                          gpu_canvas);
           
   
    // Check for errors on kernel call
    gpu_err_check(cudaGetLastError());

    // Copy the canvas back to host memory
    gpu_err_check(cudaMemcpy(canvas,
                             gpu_canvas,
                             sizeof(uint32_t) * num_pixels_x * num_pixels_z,
                             cudaMemcpyDeviceToHost));

    // Free the gpu resources
    gpu_err_check(cudaFree(gpu_canvas));
    gpu_err_check(cudaFree(gpu_spheres));
}

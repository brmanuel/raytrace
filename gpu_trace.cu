#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <stdbool.h>


#include "include/gpu_trace.hu"


__host__ __device__ uint32_t
get_idx_of_first_intersection(sphere **spheres,
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
        sphere *sphere = spheres[sphere_idx];
        vector vec_to_center = {
            ray_start.x - sphere->center.x,
            ray_start.y - sphere->center.y,
            ray_start.z - sphere->center.z,
        };
        float a = 1;
        float b = 2*(ray_x_norm * vec_to_center.x +
                     ray_y_norm * vec_to_center.y +
                     ray_z_norm * vec_to_center.z);
        float c = (vec_to_center.x * vec_to_center.x +
                   vec_to_center.y * vec_to_center.y +
                   vec_to_center.z * vec_to_center.z -
                   sphere->radius * sphere->radius);
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
gpu_trace_kernel(sphere **spheres,
                 bool *sphere_is_light,
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

    // initialize canvas to black
    memset((void *) canvas, 0, num_pixels_x * num_pixels_z * 4);

    vector zero_vec = {0.0,0.0,0.0};
    float canvas_width = canvas_max_x - canvas_min_x;
    float canvas_x_incr = canvas_width / num_pixels_x;
    float canvas_height = canvas_max_z - canvas_min_z;
    float canvas_z_incr = canvas_height / num_pixels_z;
    for (uint32_t i = 0; i < num_pixels_x; i++){
        for (uint32_t j = 0; j < num_pixels_z; j++){
            vector ray_dir = {
                canvas_min_x + (i+0.5) * canvas_x_incr,
                canvas_y,
                canvas_min_z + (j+0.5) * canvas_z_incr
            };
            vector intersection;
            uint32_t winner_sphere_idx = get_idx_of_first_intersection(spheres,
                                                                       num_spheres,
                                                                       zero_vec,
                                                                       ray_dir,
                                                                       &intersection);

            if (winner_sphere_idx < num_spheres) {
                // intersecting sphere found
                sphere *winner_sphere = spheres[winner_sphere_idx];

                // only intersection color
                canvas[i + j * num_pixels_x] = winner_sphere->color;
            }
        }
    }
}







void gpu_trace(sphere **spheres,
          bool *sphere_is_light,
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

    // Allocate GPU memory for the raw input data (either audio file
    // data or randomly generated data. The data is of type float and
    // has n_frames elements. Then copy the data in raw_data into the
    // GPU memory you allocated.
    float* gpu_raw_data;
    cudaMalloc((void **) &gpu_raw_data,
               sizeof(float) * n_frames);
    cudaMemcpy(gpu_raw_data,
               raw_data,
               sizeof(float) * n_frames,
               cudaMemcpyHostToDevice);
    
    // Allocate GPU memory for the impulse signal (for now global GPU
    // memory is fine. The data is of type float and has blur_v_size
    // elements. Then copy the data in blur_v into the GPU memory you
    // allocated.
    float* gpu_blur_v;
    cudaMalloc((void **) &gpu_blur_v,
               sizeof(float) * blur_v_size);
    cudaMemcpy(gpu_blur_v,
               blur_v,
               sizeof(float) * blur_v_size,
               cudaMemcpyHostToDevice);

    // Allocate GPU memory to store the output audio signal after the
    // convolution. The data is of type float and has n_frames elements.
    // Initialize the data as necessary.
    // TODO: do I have to initialize anything?
    float* gpu_out_data;
    cudaMalloc((void **) &gpu_out_data,
               sizeof(float) * n_frames);

    // Appropriately call the kernel function.
    cuda_blur_kernel<<<blocks, threads_per_block>>>(gpu_raw_data,
                                                    gpu_blur_v,
                                                    gpu_out_data,
                                                    n_frames,
                                                    blur_v_size);
           
   
    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

    // Now that kernel calls have finished, copy the output signal
    // back from the GPU to host memory. (We store this channel's result
    // in out_data on the host.)
    cudaMemcpy(out_data,
               gpu_out_data,
               sizeof(float) * n_frames,
               cudaMemcpyDeviceToHost);

    // Now that we have finished our computations on the GPU, free the
    // GPU resources.
    cudaFree(gpu_raw_data);
    cudaFree(gpu_blur_v);
    cudaFree(gpu_out_data);

}

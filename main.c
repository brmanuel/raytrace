#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <stdbool.h>

#include "include/write_image.h"
#include "include/rgb.h"


typedef struct _vector {
    float x;
    float y;
    float z;
} vector;

typedef struct _sphere {
    vector center;
    float radius;
    uint32_t color; // RGB as ________RRRRRRRRGGGGGGGGBBBBBBBB 
} sphere;


sphere *
create_sphere(float center_x,
              float center_y,
              float center_z,
              float radius,
              uint32_t color)
{
    sphere *s = (sphere *) malloc(sizeof(sphere));
    s->center.x = center_x;
    s->center.y = center_y;
    s->center.z = center_z;
    s->radius = radius;
    s->color = color;
    return s;
}


uint32_t
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
        float b = -2*(ray_x_norm * vec_to_center.x +
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
        if (x1 > 0 && x1 < min_dist){
            min_dist = x1;
            winner_idx = sphere_idx;
            *intersection = {
                ray_start.x + x1 * ray_x_norm,
                ray_start.y + x1 * ray_y_norm,
                ray_start.z + x1 * ray_z_norm,
            };
        }
        else if (x2 > 0 && x2 < min_dist){ // x2 must be > x1!
            min_dist = x2;
            winner_idx = sphere_idx;
            *intersection = {
                ray_start.x + x2 * ray_x_norm,
                ray_start.y + x2 * ray_y_norm,
                ray_start.z + x2 * ray_z_norm,
            };
        }
    }
    return winner_idx;
}
                       

void
trace(sphere **spheres,
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

                // only intersection color
                // canvas[i + j * num_pixels_x] = spheres[winner_sphere_idx]->color;

                // hard shadows
                uint32_t color = 0;
                for (uint32_t sphere_idx = 0; sphere_idx < num_spheres; sphere_idx++){
                    if (sphere_is_light[sphere_idx]){
                        vector light_ray_dir = {
                            spheres[sphere_idx]->center.x - intersection.x,
                            spheres[sphere_idx]->center.y - intersection.y,
                            spheres[sphere_idx]->center.z - intersection.z,
                        };
                        vector intersection_with_light;
                        uint32_t blocker_idx = get_idx_of_first_intersection(spheres,
                                                                             num_spheres,
                                                                             intersection,
                                                                             light_ray_dir,
                                                                             &intersection_with_light);

                        assert(blocker_idx < num_spheres);
                        if (blocker_idx == sphere_idx) {
                            // no object blocking the light source
                            // TODO
                            
                        }
                    }
                }
            }
        }
    }
}


int
main()
{
    uint32_t num_pixels_x = 500;
    uint32_t num_pixels_z = 500;
    uint32_t *canvas = (uint32_t *) malloc(num_pixels_x * num_pixels_z * sizeof(uint32_t));
    sphere *spheres[] = {
        create_sphere(0, 1000, 0, 985, cram_rgb(180,180,180)),
        create_sphere(1000, 10, 0, 995, cram_rgb(0,100,220)),
        create_sphere(-1000, 10, 0, 995, cram_rgb(220,100,0)),
        create_sphere(0, 10, 1000, 995, cram_rgb(130,130,130)),
        create_sphere(0, 10, -1000, 995, cram_rgb(130,130,130)),
        create_sphere(-2, 12, 3, 2, cram_rgb(200,200,200)),
        create_sphere(3, 10, 3, 2, cram_rgb(200,200,200)),
        create_sphere(0, 12, -5, 2, cram_rgb(255,255,255)),
    };
    
    trace(spheres,
          8,
          -3.0,
          3.0,
          -3.0,
          3.0,
          5.0,
          num_pixels_x,
          num_pixels_z,
          canvas);

    save_png_to_file(canvas, num_pixels_x, num_pixels_z, "outs/outfile.png");
    
    return 0;
}

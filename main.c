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

uint32_t min(uint32_t a, uint32_t b)
{
    if (a < b) {
        return a;
    }
    return b;
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
        if (x1 > 0.1 && x1 < min_dist){
            min_dist = x1;
            winner_idx = sphere_idx;
            *intersection = (vector){
                ray_start.x + x1 * ray_x_norm,
                ray_start.y + x1 * ray_y_norm,
                ray_start.z + x1 * ray_z_norm,
            };
        }
        else if (x2 > 0.1 && x2 < min_dist){ // x2 must be > x1!
            min_dist = x2;
            winner_idx = sphere_idx;
            *intersection = (vector){
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
                sphere *winner_sphere = spheres[winner_sphere_idx];

                // only intersection color
                //canvas[i + j * num_pixels_x] = winner_sphere->color;

                // hard shadows
                uint32_t color = 0;
                if (sphere_is_light[winner_sphere_idx]){
                    color = cram_rgb(255,255,255);
                }
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
                            vector normal_at_intersection = {
                                intersection.x - winner_sphere->center.x,
                                intersection.y - winner_sphere->center.y,
                                intersection.z - winner_sphere->center.z,
                            };
                            float cos_of_light_impact =
                                (normal_at_intersection.x * light_ray_dir.x +
                                 normal_at_intersection.y * light_ray_dir.y +
                                 normal_at_intersection.z * light_ray_dir.z) /
                                (sqrt(normal_at_intersection.x * normal_at_intersection.x +
                                      normal_at_intersection.y * normal_at_intersection.y +
                                      normal_at_intersection.z * normal_at_intersection.z) *
                                 sqrt(light_ray_dir.x * light_ray_dir.x +
                                      light_ray_dir.y * light_ray_dir.y +
                                      light_ray_dir.z * light_ray_dir.z));
                            if (cos_of_light_impact < 0) {
                                cos_of_light_impact = 0;
                            }
                            uint32_t new_r = min(255,
                                                 (uint32_t) uncram_rgb(color, 'r') +
                                                 cos_of_light_impact *
                                                 uncram_rgb(winner_sphere->color, 'r'));
                            uint32_t new_g = min(255,
                                                 (uint32_t) uncram_rgb(color, 'g') +
                                                 cos_of_light_impact *
                                                 uncram_rgb(winner_sphere->color, 'g'));
                            uint32_t new_b = min(255,
                                                 (uint32_t) uncram_rgb(color, 'b') +
                                                 cos_of_light_impact *
                                                 uncram_rgb(winner_sphere->color, 'b'));
                            
                            color = cram_rgb(new_r, new_g, new_b);
                        }
                    }
                }
                canvas[i + j * num_pixels_x] = color;
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
        create_sphere(0, 1000, 0, 975, cram_rgb(180,180,180)),
        create_sphere(1000, 10, 0, 992, cram_rgb(0,100,220)),
        create_sphere(-1000, 10, 0, 992, cram_rgb(220,100,0)),
        create_sphere(0, 10, 1000, 995, cram_rgb(130,130,130)),
        create_sphere(0, 10, -1000, 995, cram_rgb(130,130,130)),
        create_sphere(-2, 20, 3, 2, cram_rgb(200,200,200)),
        create_sphere(3, 10, 3, 2, cram_rgb(200,200,200)),
        create_sphere(0, 12, -4, 1, cram_rgb(255,255,255)),
    };
    bool sphere_is_light[] = {
        false, false, false, false, false, false, false, true
    };
    trace(spheres,
          sphere_is_light,
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

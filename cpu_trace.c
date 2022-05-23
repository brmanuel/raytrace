#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <stdbool.h>


#include "include/cpu_trace.h"
#include "include/rgb.h"


/* given the light_ray hitting the object,
   the normal at the intersection between object and light_ray,
   and the color of the object,
   computes the lambert luminescence of the light at the object.
 */
static uint32_t
get_lambert_color_increment(vector light_ray,
                            vector object_normal,
                            uint32_t object_color)
{

    float cos_of_light_impact =
        (object_normal.x * light_ray.x +
         object_normal.y * light_ray.y +
         object_normal.z * light_ray.z) /
        (sqrt(object_normal.x * object_normal.x +
              object_normal.y * object_normal.y +
              object_normal.z * object_normal.z) *
         sqrt(light_ray.x * light_ray.x +
              light_ray.y * light_ray.y +
              light_ray.z * light_ray.z));
    /* account for numeric imprecision */
    if (cos_of_light_impact < 0) {
        cos_of_light_impact = 0;
    }
    return cram_rgb(cos_of_light_impact * uncram_rgb(object_color, 'r'),
                    cos_of_light_impact * uncram_rgb(object_color, 'g'),
                    cos_of_light_impact * uncram_rgb(object_color, 'b'));
}


static uint32_t
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
                       

void
cpu_trace(sphere **spheres,
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
                            uint32_t color_increment =
                                get_lambert_color_increment(normal_at_intersection,
                                                            light_ray_dir,
                                                            winner_sphere->color);
                            
                            color = add_crammed_rgb(color, color_increment);
                        }
                    }
                }
                canvas[i + j * num_pixels_x] = color;
            }
        }
    }
}

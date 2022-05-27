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
#include "include/structs.h"
#include "include/cpu_trace.h"


sphere *
create_sphere(float center_x,
              float center_y,
              float center_z,
              float radius,
              uint32_t color,
              bool is_light)
{
    sphere *s = (sphere *) malloc(sizeof(sphere));
    s->center.x = center_x;
    s->center.y = center_y;
    s->center.z = center_z;
    s->radius = radius;
    s->color = color;
    s->is_light = is_light;
    return s;
}



int
main()
{
    uint32_t num_pixels_x = 500;
    uint32_t num_pixels_z = 500;
    uint32_t *canvas = (uint32_t *) malloc(num_pixels_x * num_pixels_z * sizeof(uint32_t));
    sphere *spheres[] = {
        create_sphere(0, 1000, 0, 975, cram_rgb(180,180,180), false),
        create_sphere(1000, 10, 0, 992, cram_rgb(0,100,220), false),
        create_sphere(-1000, 10, 0, 992, cram_rgb(220,100,0), false),
        create_sphere(0, 10, 1000, 995, cram_rgb(130,130,130), false),
        create_sphere(0, 10, -1000, 995, cram_rgb(130,130,130), false),
        create_sphere(-2, 20, 3, 2, cram_rgb(200,200,200), false),
        create_sphere(3, 10, 3, 2, cram_rgb(200,200,200), false),
        create_sphere(0, 12, -4, 1, cram_rgb(255,255,255), true),
    };
    cpu_trace(spheres,
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

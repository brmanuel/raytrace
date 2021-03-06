#pragma once

#include "structs.h"

/* Raytrace the scene on the cpu. */
void
cpu_trace(sphere **spheres,
          uint32_t num_spheres,
          float canvas_min_x,
          float canvas_max_x,
          float canvas_min_z,
          float canvas_max_z,
          float canvas_y,
          uint32_t num_pixels_x,
          uint32_t num_pixels_z,
          uint32_t *canvas);

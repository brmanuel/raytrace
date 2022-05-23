#pragma once
#include <stdint.h>

/* store rgb in a single uint32_t */
uint32_t
cram_rgb(uint32_t r, uint32_t g, uint32_t b);

/* extract r, g, and b from a uint32_t */
uint32_t
uncram_rgb(uint32_t rgb, char to_extract);

/* add two rgb colors component-wise, topping at 255. */
uint32_t
add_crammed_rgb(uint32_t color1, uint32_t color2);

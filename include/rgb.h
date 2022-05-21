#pragma once
#include <stdint.h>

uint32_t
cram_rgb(uint32_t r, uint32_t g, uint32_t b);

uint32_t
uncram_rgb(uint32_t rgb, char to_extract);

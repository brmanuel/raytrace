#include "include/rgb.h"

uint32_t
cram_rgb(uint32_t r,
         uint32_t g,
         uint32_t b)
{
    uint32_t bytemask = (1 << 8) - 1;
    return (((r & bytemask) << 16) |
            ((g & bytemask) << 8) |
            ((b & bytemask) << 0));
}

uint32_t
uncram_rgb(uint32_t rgb,
           char to_extract)
{
    uint32_t bytemask = (1 << 8) - 1;
    switch (to_extract) {
    case 'r':
        return (rgb >> 16) & bytemask;
    case 'g':
        return (rgb >> 8) & bytemask;
    case 'b':
        return (rgb >> 0) & bytemask;
    default:
        // error case, return invalid byte
        return -1;        
    }
}



static uint32_t
min(uint32_t a, uint32_t b)
{
    if (a < b) {
        return a;
    }
    return b;
}


uint32_t
add_crammed_rgb(uint32_t color1,
                uint32_t color2)
{
    uint32_t new_r = min(255,
                         (uint32_t) uncram_rgb(color1, 'r') +
                         (uint32_t) uncram_rgb(color2, 'r'));
    uint32_t new_g = min(255,
                         (uint32_t) uncram_rgb(color1, 'g') +
                         (uint32_t) uncram_rgb(color2, 'g'));
    uint32_t new_b = min(255,
                         (uint32_t) uncram_rgb(color1, 'b') +
                         (uint32_t) uncram_rgb(color2, 'b'));
    return cram_rgb(new_r, new_g, new_b);

}

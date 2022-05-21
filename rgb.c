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

#include <png.h>
#include <stdlib.h>
#include <stdint.h>

#include "include/write_image.h"
#include "include/rgb.h"

int
save_png_to_file (uint32_t *canvas,
                  uint32_t canvas_width,
                  uint32_t canvas_height,
                  const char *path)
{
    FILE * fp;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    size_t x, y;
    png_byte ** row_pointers = NULL;
    /* "status" contains the return value of this function. At first
       it is set to a value which means 'failure'. When the routine
       has finished its work, it is set to a value which means
       'success'. */
    int status = -1;
    /* The following number is set by trial and error only. I cannot
       see where it it is documented in the libpng manual.
    */
    int pixel_size = 3;
    int depth = 8;
    
    fp = fopen (path, "wb");
    if (! fp) {
        return status;
    }

    png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        fclose(fp);
        return status;
    }
    
    info_ptr = png_create_info_struct (png_ptr);
    if (info_ptr == NULL) {        
        png_destroy_write_struct (&png_ptr, &info_ptr);
        fclose(fp);
        return status;
    }
    
    /* Set up error handling. */

    if (setjmp (png_jmpbuf (png_ptr))) {        
        png_destroy_write_struct (&png_ptr, &info_ptr);
        fclose(fp);
        return status;
    }
    
    /* Set image attributes. */

    png_set_IHDR (png_ptr,
                  info_ptr,
                  canvas_width,
                  canvas_height,
                  depth,
                  PNG_COLOR_TYPE_RGB,
                  PNG_INTERLACE_NONE,
                  PNG_COMPRESSION_TYPE_DEFAULT,
                  PNG_FILTER_TYPE_DEFAULT);
    
    /* Initialize rows of PNG. */

    row_pointers = png_malloc (png_ptr, canvas_height * sizeof (png_byte *));
    for (y = 0; y < canvas_height; y++) {
        png_byte *row = png_malloc (png_ptr, sizeof (uint8_t) * canvas_width * pixel_size);
        row_pointers[y] = row;
        for (x = 0; x < canvas_width; x++) {
            *row++ = uncram_rgb(canvas[x + y * canvas_width], 'r');
            *row++ = uncram_rgb(canvas[x + y * canvas_width], 'g');
            *row++ = uncram_rgb(canvas[x + y * canvas_width], 'b');
        }
    }
    
    /* Write the image data to "fp". */

    png_init_io (png_ptr, fp);
    png_set_rows (png_ptr, info_ptr, row_pointers);
    png_write_png (png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    /* The routine has successfully written the file, so we set
       "status" to a value which indicates success. */

    status = 0;
    
    for (y = 0; y < canvas_height; y++) {
        png_free (png_ptr, row_pointers[y]);
    }
    png_free (png_ptr, row_pointers);
    
}

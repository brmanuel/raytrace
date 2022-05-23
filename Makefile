IDIR =include
CC=gcc
CFLAGS=-g -I$(IDIR)
LIBS=-lm -lpng

_DEPS = rgb.h write_image.h structs.h cpu_trace.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

ODIR = obj
_OBJ = main.o write_image.o rgb.o cpu_trace.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

raytracer: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)


.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core

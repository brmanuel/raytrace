## About
Simple raytracer. First sequential implementation. Goal is experimenting with CUDA.

## Example images
Example images, tracking versions. These mimick the example at [wikipedia](https://de.wikipedia.org/wiki/Raytracing).

### Version 1: Only intersection test, no light, no recursive rays.
![no lights](images/no_lights.png "No Lights")

### Version 2: simple lighting, lambert shader, no recursive rays.
![lambert](images/lambert_shader.png "Lambert")


## TODO
- test initial cuda implementation (2h)
  - does it run on device
  - does it give correct result
  - is it faster than sequential version (for larger image sizes)

- extend sequential ray-tracer to diffuse and recursive raytracing (10h)
  - find some resource to read about parameters and formulae
  - implement:
    - spheres have reflection and refraction parameters
    - recursively trace rays -> soft shadows
    - parametrizable recursion-depth & diffusion (how many light-rays to send out)

- update Cuda version to same state as sequential version (4h)
  - Risk: can we do recursion as desired (should work out of the box)

- build infrastructure to run different versions of code (4h)
  - refactor to make it easy to write multiple versions of code to enable code re-use
  - make different cuda versions selectable conveniently
  - compare result with golden model
  - measure execution time
  - setup a small benchmark set: set of inputs, a script to run them all in one go (Makefile target), record some KPIs across implementation versions

- experiment with cuda performance (10h)
  - what is peak performance we can expect
  - try out a tool to find bottlenecks / measure utilization / get some performance numbers from device
  - try out different memory types
  - try out different grids
  - read Cuda best practice guide for more things to try 
## About
Simple raytracer. First sequential implementation. Goal is experimenting with CUDA.

## Example images
Example images, tracking versions. These mimick the example at [wikipedia](https://de.wikipedia.org/wiki/Raytracing).

### Version 1: Only intersection test, no light, no recursive rays.
![no lights](images/no_lights.png "No Lights")

### Version 2: simple lighting, lambert shader, no recursive rays.
![lambert](images/lambert_shader.png "Lambert")

### Version 3 (WIP): [diffuse raytracing](https://dl.acm.org/doi/pdf/10.1145/964965.808590).
- We want to approximate the light intensity $I$ at every pixel of the image
- Every pixel in our image corresponds to a point $p$ in the "scene"
- Consider the unit sphere $\Omega(p)$ parametrized by spherical coordinates $\phi \in [0, 2\pi), \tau \in [-\frac{\pi}{2}, \frac{\pi}{2}]$ around $p$
- The illumination of $p$ is determined by the sum of light rays meeting $p$ from every direction and on the reflection/refraction of each light ray in the direction of the camera.
- Mathematically speaking we can
  - represent the intensity of a light ray meeting $p$ from each direction $(\phi, \tau)$ by a function $L(\phi, \tau)$
  - represent the reflection of light ray from each direction $(\phi, \tau)$ "above" the surface by a function $R(\phi, \tau)$ which depends on the direction of the camera.
  - represent the refraction (or transmittance) of light rays from each direction $(\phi, \tau)$ "below" the surface by a function $T(\phi, \tau)$ which depends on the direction of the camera.
  - integrate the product of $L \cdot (R + T)$ over the whole sphere.
- To approximate the integral we perform monte-carlo sampling:
  - For reflections we sample rays according to the specular reflectance function $R$ 
  - For refraction we sample rays according to the specular transmittance function $T$
  - For shadows we sample rays according to the intensity of the light source in that direction
- Each of these sampling functions can probably best be represented by a normal where sigma corresponds to "fuzziness".

- Q: How to model different surface properties: translucency, reflection, color
- Q: How to best "distribute" secondary rays over the different purposes? 


## TODO
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

## Potential further steps
- raytracer extensions:
  - support triangles as shapes
  - depth of field by simulating a lense
  - optimization of intersection test: aggregating objects hierarchically
  - full path tracing: sample the render equation

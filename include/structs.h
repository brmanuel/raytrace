#pragma once


/* a 3d vector */
typedef struct _vector {
    float x;
    float y;
    float z;
} vector;

/* the basic element with which we compose our scenes. */
typedef struct _sphere {
    vector center;
    float radius;
    uint32_t color; // RGB as ________RRRRRRRRGGGGGGGGBBBBBBBB 
} sphere;

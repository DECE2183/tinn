#pragma once

#include <stdio.h>
#include <stdint.h>

typedef struct
{
    float  value;
    float  bias;
    float* weights;
}
neuron_t;

typedef struct
{
    uint32_t   layers_count;
    uint32_t*  layers_sizes;
    neuron_t** neurons;
    float*     output;
}
Tinn;

float* xtpredict(Tinn, const float* in);

float xttrain(Tinn, const float* in, const float* tg, float rate);

Tinn xtbuild(const uint32_t layers_count, const uint32_t *layers_sizes);

void xtsave(Tinn, FILE *const file);

Tinn xtload(FILE *const file);

void xtfree(Tinn);

void xtprint(const float* arr, const int32_t size);

#include "Tinn.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Computes error.
static inline float err(const float a, const float b)
{
    return 0.5f * (a - b) * (a - b);
}

// Returns partial derivative of error function.
static inline float pderr(const float a, const float b)
{
    return a - b;
}

// Computes total error of target to output.
static inline float toterr(const float *const tg, const neuron_t *const o, const int32_t size)
{
    float sum = 0.0f;
    for(int i = 0; i < size; ++i)
        sum += err(tg[i], o[i].value);
    return sum;
}

// Activation function.
static inline float act(const float a)
{
    return 1.0f / (1.0f + expf(-a));
}

// Returns partial derivative of activation function.
static inline float pdact(const float a)
{
    return a * (1.0f - a);
}

// Returns floating point random from 0.0 - 1.0.
static inline float frand()
{
    return rand() / (float) RAND_MAX;
}

// Calculate neuron values by passing input as float array
static inline void fcalcin(const float *const in, const uint32_t in_len, neuron_t *const out, const uint32_t out_len)
{
    int i, j;
    float sum;

    for (i = 0; i < out_len; ++i)
    {
        sum = 0.0f;
        for (j = 0; j < in_len; ++j)
        {
            sum += in[j] * out[i].weights[j];
        }
        out[i].value = act(sum + out[i].bias);
    }
}

// Calculate neuron values by passing input as neurons
static inline void fcalclayer(const neuron_t *const in, const uint32_t in_len, neuron_t *const out, const uint32_t out_len)
{
    int i, j;
    float sum;

    for (i = 0; i < out_len; ++i)
    {
        sum = 0.0f;
        for (j = 0; j < in_len; ++j)
        {
            sum += in[j].value * out[i].weights[j];
        }
        out[i].value = act(sum + out[i].bias);
    }
}

// Performs back propagation.
static void bprop(Tinn t, const float* const in, const float* const tg, float rate)
{
    int i, j, k;
    float sum = 0.0f;
    float a, b;

    neuron_t* layer = t.neurons[t.layers_count-2];
    for (i = 0; i < t.layers_sizes[t.layers_count-1]; ++i)
    {
        a = pderr(layer[i].value, tg[i]);
        b = pdact(layer[i].value);

        for (j = 0; j < t.layers_sizes[t.layers_count-2]; ++j)
        {
            sum += a * b * layer[i].weights[j];
            // Correct weights in hidden to output layer.
            layer[i].weights[j] -= rate * a * b * t.neurons[t.layers_count-3][j].value;
        }

        layer[i].bias -= rate * a * b * 0.5;
    }

    for (i = t.layers_count-3; i > 0; --i) // per each hidden layer
    {
        layer = t.neurons[i];
        for (j = 0; j < t.layers_sizes[i+1]; ++j) // per current hidden layer neurons
        {
            b = pdact(layer[j].value);
            for (k = 0; k < t.layers_sizes[i]; ++k) // per previous hidden layer neurons
            {
                layer[j].weights[k] -= rate * sum * b * t.neurons[i-1][k].value;
            }
            layer[j].bias -= rate * sum * b * 0.5;
        }
    }

    layer = t.neurons[0];
    for (i = 0; i < t.layers_sizes[1]; ++i) // per first hidden layer neurons
    {
        b = pdact(layer[i].value);
        for (j = 0; j < t.layers_sizes[0]; ++j) // per input layer neurons
        {
            layer[i].weights[j] -= rate * sum * b * in[j];
        }
        layer[i].bias -= rate * sum * b * 0.5;
    }
}

// Performs forward propagation.
static inline void fprop(Tinn t, const float* const in)
{
    // Calculate hidden layer neuron values.
    fcalcin(in, t.layers_sizes[0], t.neurons[0], t.layers_sizes[1]);
    // Calculate other layer neuron values.
    for (int i = 0; i < t.layers_count-2; ++i)
    {
        fcalclayer(t.neurons[i], t.layers_sizes[i+1], t.neurons[i+1], t.layers_sizes[i+2]);
    }
}

// Randomizes tinn weights and biases.
static void wbrand(Tinn t)
{
    for (int i = 0; i < t.layers_count-1; ++i)
    {
        for (int j = 0; j < t.layers_sizes[i+1]; ++j)
        {
            for (int k = 0; k < t.layers_sizes[i]; ++k)
            {
                t.neurons[i][j].weights[k] = frand()*2.0f - 1.0f;
                t.neurons[i][j].bias = frand()*2.0f - 1.0f;
            }
        }
    }
}

// Returns an output prediction given an input.
float* xtpredict(Tinn t, const float* const in)
{
    fprop(t, in);
    for (int i = 0; i < t.layers_sizes[t.layers_count-1]; ++i)
    {
        t.output[i] = t.neurons[t.layers_count-2][i].value;
    }
    return t.output;
}

// Trains a tinn with an input and target output with a learning rate. Returns target to output error.
float xttrain(Tinn t, const float* const in, const float* const tg, float rate)
{
    fprop(t, in);
    bprop(t, in, tg, rate);
    return toterr(tg, t.neurons[t.layers_count-2], t.layers_sizes[t.layers_count-1]);
}

// Constructs a tinn with number of inputs, number of hidden neurons, and number of outputs
Tinn xtbuild(const uint32_t layers_count, const uint32_t *layers_sizes)
{
    Tinn t = {0};
    t.layers_count = layers_count;
    t.layers_sizes = calloc(layers_count, sizeof(float));
    memcpy(t.layers_sizes, layers_sizes, sizeof(uint32_t) * layers_count);

    t.neurons = calloc(layers_count-1, sizeof(neuron_t*));
    for (int i = 0; i < layers_count-1; ++i)
    {
        t.neurons[i] = calloc(layers_sizes[i+1], sizeof(neuron_t));
        for (int j = 0; j < t.layers_sizes[i+1]; ++j)
        {
            t.neurons[i][j].weights = calloc(sizeof(float), t.layers_sizes[i]);
        }
    }

    t.output = calloc(layers_sizes[layers_count+1], sizeof(float));
    wbrand(t);
    return t;
}

// Saves a tinn to disk.
void xtsave(Tinn t, FILE* const file)
{
    // save layers count
    fwrite(&t.layers_count, sizeof(uint32_t), 1, file);

    // save sizes of layers
    fwrite(t.layers_sizes, sizeof(uint32_t), t.layers_count, file);

    // save weights and biases
    for (int i = 0; i < t.layers_count-1; ++i)
    {
        for (int j = 0; j < t.layers_sizes[i+1]; ++j)
        {
            fwrite(t.neurons[i][j].weights, sizeof(float), t.layers_sizes[i], file);
            fwrite(&t.neurons[i][j].bias, sizeof(float), 1, file);
        }
    }
}

// Loads a tinn from disk.
Tinn xtload(FILE *const file)
{
    // load layers count
    uint32_t layers_count;
    fread(&layers_count, sizeof(uint32_t), 1, file);

    // load layers sizes
    uint32_t* layers_sizes = calloc(layers_count, sizeof(uint32_t));
    fread(layers_sizes, sizeof(uint32_t) * layers_count, 1, file);

    Tinn t = xtbuild(layers_count, layers_sizes);

    // load weights and biases
    for (int i = 0; i < t.layers_count - 1; ++i)
    {
        for (int j = 0; j < t.layers_sizes[i+1]; ++j)
        {
            fread(t.neurons[i][j].weights, sizeof(float), t.layers_sizes[i], file);
            fread(&t.neurons[i][j].bias, sizeof(float), 1, file);
        }
    }

    free(layers_sizes);
    return t;
}

// Frees object from heap.
void xtfree(Tinn t)
{
    for (int i = 0; i < t.layers_count-1; ++i)
    {
        for (int j = 0; j < t.layers_sizes[i+1]; ++j)
        {
            free(t.neurons[i][j].weights);
        }
        free(t.neurons[i]);
    }

    free(t.output);
    free(t.layers_sizes);
}

// Prints an array of floats. Useful for printing predictions.
void xtprint(const float* arr, const int32_t size)
{
    for(int32_t i = 0; i < size; ++i)
        printf("%f ", (double) arr[i]);
    printf("\n");
}

#ifndef LAYER_H
#define LAYER_H

#include <cstdio>
#include "cnpy.h"
#include <cstdlib>

struct Layer {

	bool init;

    std::string network;

    std::string name;

    std::string type;

    bool ReLU;

    int stride;

    int padding;

    /* numpy array containing the weights for the layer */
    float* weights;
    std::vector<size_t> wgt_shape;

    /* numpy array containing the bias for the layer */
    float* bias;
    std::vector<size_t> bias_shape;

    /* numpy array containing the activations for the layer */
    float* activations;
    std::vector<size_t> act_shape;

    /* numpy array containing the output activations for the layer */
    float* output_activations;
    std::vector<size_t> out_act_shape;

    Layer(const std::string &_network, const std::string &_name, const std::string &_type, bool _ReLU, int _stride,
            int _padding);
    ~Layer();

    float act_get(int i, int j, int k, int l) const;
    float wgt_get(int i, int j, int k, int l) const;
    uint64_t getMaxIndex(const std::string &array) const;
    void zero_pad();
    void act_split_4D(int K, int X, int Y);
    void wgt_split_4D(int K, int X, int Y);
    void reshape_to_2D();
    void read_layer();

};
#endif

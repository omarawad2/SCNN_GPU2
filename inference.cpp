
// Includes

#include "cnpy.h"
#include <math.h>

// Data structures

struct Layer {

    std::string network = "";

    std::string name = "";

    std::string type = "";

    bool ReLU = false;

    int stride = 1;

    int padding = 0;

    /* numpy array containing the weights for the layer */
    float* weights = nullptr;
    std::vector<size_t> wgt_shape;

    /* numpy array containing the bias for the layer */
    float* bias = nullptr;
    std::vector<size_t> bias_shape;

    /* numpy array containing the activations for the layer */
    float* activations = nullptr;
    std::vector<size_t> act_shape;

    /* numpy array containing the output activations for the layer */
    float* output_activations = nullptr;
    std::vector<size_t> out_act_shape;

    Layer(const std::string &_network, const std::string &_name, const std::string &_type, bool _ReLU, int _stride,
            int _padding) : ReLU(_ReLU), stride(_stride), padding(_padding) {
        this->network = _network;
        this->name = _name;
        this->type = _type;

    }

    ~Layer() {
        free(weights);
        free(bias);
        free(activations);
        free(output_activations);
    }

    float act_get(int i, int j, int k, int l) {
        auto index = act_shape[1]*act_shape[2]*act_shape[3]*i + act_shape[2]*act_shape[3]*j + act_shape[3]*k + l;
        return activations[index];
    }

    float wgt_get(int i, int j, int k, int l) {
        auto index = wgt_shape[1]*wgt_shape[2]*wgt_shape[3]*i + wgt_shape[2]*wgt_shape[3]*j + wgt_shape[3]*k + l;
        return weights[index];
    }

    uint64_t getMaxIndex(const std::string &array) const {
        if(array == "weights") {
            return wgt_shape[0]*wgt_shape[1]*wgt_shape[2]*wgt_shape[3];
        } else if(array == "bias") {
            return bias_shape[0];
        } else if(array == "activations") {
            return act_shape[0]*act_shape[1]*act_shape[2]*act_shape[3];
        } else if(array == "output_activations") {
            if(out_act_shape.size() == 4) return out_act_shape[0]*out_act_shape[1]*out_act_shape[2]*out_act_shape[3];
            else return out_act_shape[0]*out_act_shape[1];
        } else return 0;
    }

    void zero_pad() {

        auto batch_size = act_shape[0];
        auto act_channels = act_shape[1];
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];
        auto new_Nx = Nx + 2*padding;
        auto new_Ny = Ny + 2*padding;

        uint64_t new_max_index = batch_size * act_channels * new_Nx * new_Ny;
        auto tmp_activations = (float *) malloc(new_max_index * sizeof(float));
        if (tmp_activations == nullptr) {
            fprintf(stderr, "Error: Failed to allocate padded activations!\n");
            exit(EXIT_FAILURE);
        }

        for(uint64_t i = 0; i < new_max_index; i++) {
            tmp_activations[i] = 0;
        }

        for(int n = 0; n < batch_size; n++) {
            for (int k = 0; k < act_channels; k++) {
                for (int i = 0; i < Nx; i++) {
                    for(int j = 0; j < Ny; j++) {
                        auto index_out = act_channels*new_Nx*new_Ny*n + new_Nx*new_Ny*k + new_Ny*(padding + i) +
                                (padding + j);
                        auto index_in = act_channels*Nx*Ny*n + Nx*Ny*k + Ny*i + j;
                        tmp_activations[index_out] = activations[index_in];
                    }
                }
            }
        }

        free(activations);
        activations = tmp_activations;
        act_shape.clear();
        act_shape.push_back(batch_size);
        act_shape.push_back(act_channels);
        act_shape.push_back(new_Nx);
        act_shape.push_back(new_Ny);

    }

    void reshape_to_2D() {

        auto batch_size = act_shape[0];
        auto act_channels = act_shape[1];
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];
        auto new_act_channels = act_channels * Nx * Ny;

        act_shape.clear();
        act_shape.push_back(batch_size);
        act_shape.push_back(new_act_channels);
        act_shape.push_back(1);
        act_shape.push_back(1);

    }

};

// Read network from numpy arrays

void read_layer(Layer &layer) {

    cnpy::NpyArray data_npy;
    uint64_t max_index;

    cnpy::npy_load("net_traces/" + layer.network + "/wgt-" + layer.name + ".npy" , data_npy, layer.wgt_shape);
    max_index = layer.getMaxIndex("weights");
    layer.weights = (float *) malloc(max_index * sizeof(float));
    if (layer.weights == nullptr) {
        fprintf(stderr, "Error: Failed to allocate weights!\n");
        exit(EXIT_FAILURE);
    }
    for(uint32_t i = 0; i < max_index; i++)
        layer.weights[i] = data_npy.data<float>()[i];

    cnpy::npy_load("net_traces/" + layer.network + "/bias-" + layer.name + ".npy" , data_npy, layer.bias_shape);
    max_index = layer.getMaxIndex("bias");
    layer.bias = (float *) malloc(max_index * sizeof(float));
    if (layer.bias == nullptr) {
        fprintf(stderr, "Error: Failed to allocate bias!\n");
        exit(EXIT_FAILURE);
    }
    for(uint32_t i = 0; i < max_index; i++)
        layer.bias[i] = data_npy.data<float>()[i];

    cnpy::npy_load("net_traces/" + layer.network + "/act-" + layer.name + "-0.npy" , data_npy, layer.act_shape);
    max_index = layer.getMaxIndex("activations");
    layer.activations = (float *) malloc(max_index * sizeof(float));
    if (layer.activations == nullptr) {
        fprintf(stderr, "Error: Failed to allocate activations!\n");
        exit(EXIT_FAILURE);
    }
    for(uint32_t i = 0; i < max_index; i++)
        layer.activations[i] = data_npy.data<float>()[i];

    cnpy::npy_load("net_traces/" + layer.network + "/act-" + layer.name + "-0-out.npy" , data_npy, layer.out_act_shape);
    max_index = layer.getMaxIndex("output_activations");
    layer.output_activations = (float *) malloc(max_index * sizeof(float));
    if (layer.output_activations == nullptr) {
        fprintf(stderr, "Error: Failed to allocate output activations!\n");
        exit(EXIT_FAILURE);
    }
    for(uint32_t i = 0; i < max_index; i++)
        layer.output_activations[i] = data_npy.data<float>()[i];

    printf("Layer %s loaded into memory\n",layer.name.c_str());

}

std::vector<Layer> read_bvlc_alexnet() {
    std::vector<Layer> network;
    //network.emplace_back(Layer("bvlc_alexnet","conv1","conv",true,4,0));
    //network.emplace_back(Layer("bvlc_alexnet","conv2","conv",true,1,2));
    //network.emplace_back(Layer("bvlc_alexnet","conv3","conv",true,1,1));
    //network.emplace_back(Layer("bvlc_alexnet","conv4","conv",true,1,1));
    network.emplace_back(Layer("bvlc_alexnet","conv5","conv",true,1,1));
    //network.emplace_back(Layer("bvlc_alexnet","fc6","fc",true,1,0));
    //network.emplace_back(Layer("bvlc_alexnet","fc7","fc",true,1,0));
    //network.emplace_back(Layer("bvlc_alexnet","fc8","fc",false,1,0));
    return network;
};

// Auxiliary functions

static inline float ReLU(const float &value) {
    return value < 0 ? 0 : value;
}

// Check function

void check_values(const Layer &layer, const float* output_activations, float min_error = 0.01) {

    printf("Checking values for layer: %s of type %s\n",layer.name.c_str(),layer.type == "conv" ? "convolution" :
            "fully connected");
    uint32_t count = 0;
    for(uint32_t i = 0; i < layer.getMaxIndex("output_activations"); i++) {
        if(fabsf(output_activations[i] - layer.output_activations[i]) > min_error) count++;
    }
    printf("ERRORS: %u out of %lu with absolute error tolerance of %.2f\n\n",count,
            layer.getMaxIndex("output_activations"), min_error);
}

// MAIN

int main(int argc, char *argv[]) {

    auto network = read_bvlc_alexnet();

    for(auto layer : network) {

        read_layer(layer);

        if(layer.type == "conv") {

            int batch_size = (int) layer.act_shape[0];
            int act_channels = (int) layer.act_shape[1];
            int Nx = (int) layer.act_shape[2];
            int Ny = (int) layer.act_shape[3];

            int num_filters = (int) layer.wgt_shape[0];
            int wgt_channels = (int) layer.wgt_shape[1];
            int Kx = (int) layer.wgt_shape[2];
            int Ky = (int) layer.wgt_shape[3];

            int padding = layer.padding;
            int stride = layer.stride;

            layer.zero_pad();
            long out_x = (Nx - Kx + 2 * padding) / stride + 1;
            long out_y = (Ny - Ky + 2 * padding) / stride + 1;

            int groups = act_channels / wgt_channels;
            int it_per_group = num_filters / groups;

            // Initialize variables
            std::vector<size_t> output_shape;
            output_shape.push_back((unsigned) batch_size);
            output_shape.push_back((unsigned) num_filters);
            output_shape.push_back((unsigned) out_x);
            output_shape.push_back((unsigned) out_y);

            auto output_activations = (float *) malloc(batch_size * num_filters * out_x * out_y * sizeof(float));
            if (output_activations == nullptr) {
                fprintf(stderr, "Error: Failed to allocate output activations!\n");
                exit(EXIT_FAILURE);
            }

            for (int n = 0; n < batch_size; n++) {
                int current_group = 0, group_m = 0, start_group = 0;
                for (int m = 0; m < num_filters; m++) {
                    for (int x = 0; x < out_x; x++) {
                        for (int y = 0; y < out_y; y++) {
                            float sum = layer.bias[m];
                            for (int i = 0; i < Kx; i++) {
                                for (int j = 0; j < Ky; j++) {
                                    for (int k = start_group; k < wgt_channels + start_group; k++) {
                                        sum += layer.act_get(n, k, stride * x + i, stride * y + j) *
                                               layer.wgt_get(m, k - start_group, i, j);
                                        printf("Out_Act[%d][%d][%d][%d] += Act[%d][%d][%d][%d]*Wgt[%d][%d][%d][%d]\n",
                                                n,m,x,y, n,k,stride * x + i,stride * y + j, m,k - start_group,i,j);
                                    }
                                }
                            }
                            if (layer.ReLU) sum = ReLU(sum);
                            auto pos = n * out_x * out_y * num_filters + m * out_x * out_y + x * out_y + y;
                            output_activations[pos] = sum;
                        }
                    }
                    group_m++;
                    if (group_m >= it_per_group) {
                        group_m = 0;
                        current_group++;
                        start_group = wgt_channels * current_group;
                    }
                }
            }

            check_values(layer,output_activations);

        } else if (layer.type == "fc") {

            if(layer.act_shape[2] != 1 && layer.act_shape[3] != 1) {
                layer.reshape_to_2D();
            }

            int batch_size = (int)layer.act_shape[0];
            int num_filters = (int)layer.wgt_shape[0];
            int wgt_channels = (int)layer.wgt_shape[1];

            std::vector<size_t> output_shape;
            output_shape.push_back((unsigned) batch_size);
            output_shape.push_back((unsigned) num_filters);

            auto output_activations = (float *) malloc(batch_size * num_filters * sizeof(float));
            if (output_activations == nullptr) {
                fprintf(stderr, "Error: Failed to allocate output activations!\n");
                exit(EXIT_FAILURE);
            }

            for (int n = 0; n<batch_size; n++) {
                for (int m = 0; m<num_filters; m++) {
                    float sum = layer.bias[m];
                    for (int k = 0; k<wgt_channels; k++) {
                        sum += layer.act_get(n, k, 0, 0) * layer.wgt_get(m, k, 0, 0);
                    }
                    if (layer.ReLU) sum = ReLU(sum);
                    output_activations[n*num_filters + m] = sum;
                }
            }

            check_values(layer,output_activations);
            free(output_activations);

        }

    }

    return 0;
}


// Includes

#include "cnpy.h"
#include <cmath>

// Constants

/* Number of PE per column */
const int Wt = 8;

/* Number of PE per row */
const int Ht = 8;

/* Column multipliers per PE */
const int I = 4;

/* Row multipliers per PE */
const int F = 4;

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

    float act_get(int i, int j, int k, int l) const {
        auto index = act_shape[1]*act_shape[2]*act_shape[3]*i + act_shape[2]*act_shape[3]*j + act_shape[3]*k + l;
        return activations[index];
    }

    float wgt_get(int i, int j, int k, int l) const {
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

    void grid_zero_pad(int X, int Y) {

        auto batch_size = act_shape[0];
        auto act_channels = act_shape[1];
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];

        uint64_t new_max_index = batch_size * act_channels * X * Y;
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
                        auto index_out = act_channels*X*Y*n + X*Y*k + Y*i + j;
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
        act_shape.push_back((unsigned)X);
        act_shape.push_back((unsigned)Y);

    }

    void act_split_4D(int K, int X, int Y) {

        auto batch_size = act_shape[0];
        auto act_channels = act_shape[1];
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];

        uint64_t new_max_index = batch_size * K * X * Y;
        auto tmp_activations = (float *) malloc(new_max_index * sizeof(float));
        if (tmp_activations == nullptr) {
            fprintf(stderr, "Error: Failed to allocate padded activations!\n");
            exit(EXIT_FAILURE);
        }

        for(int n = 0; n < batch_size; n++) {
            for (int k = 0; k < act_channels; k++) {
                for (int i = 0; i < Nx; i++) {
                    for(int j = 0; j < Ny; j++) {
                        auto new_k = k / (X*Y);
                        auto rem = k % (X*Y);
                        auto new_i = rem / Y;
                        auto new_j = rem % Y;
                        auto index_out = K*X*Y*n + X*Y*new_k + Y*new_i + new_j;
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
        act_shape.push_back((unsigned)K);
        act_shape.push_back((unsigned)X);
        act_shape.push_back((unsigned)Y);

    }

    void wgt_split_4D(int K, int X, int Y) {

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        uint64_t new_max_index = num_filters * K * X * Y;
        auto tmp_weights = (float *) malloc(new_max_index * sizeof(float));
        if (tmp_weights == nullptr) {
            fprintf(stderr, "Error: Failed to allocate padded weights!\n");
            exit(EXIT_FAILURE);
        }

        for(int n = 0; n < num_filters; n++) {
            for (int k = 0; k < wgt_channels; k++) {
                for (int i = 0; i < Kx; i++) {
                    for(int j = 0; j < Ky; j++) {
                        auto new_k = k / (X*Y);
                        auto rem = k % (X*Y);
                        auto new_i = rem / Y;
                        auto new_j = rem % Y;
                        auto index_out = K*X*Y*n + X*Y*new_k + Y*new_i + new_j;
                        auto index_in = wgt_channels*Kx*Ky*n + Kx*Ky*k + Ky*i + j;
                        tmp_weights[index_out] = weights[index_in];
                    }
                }
            }
        }

        free(weights);
        weights = tmp_weights;
        wgt_shape.clear();
        wgt_shape.push_back(num_filters);
        wgt_shape.push_back((unsigned)K);
        wgt_shape.push_back((unsigned)X);
        wgt_shape.push_back((unsigned)Y);

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
    network.emplace_back(Layer("bvlc_alexnet","conv1","conv",true,4,0));
    network.emplace_back(Layer("bvlc_alexnet","conv2","conv",true,1,2));
    network.emplace_back(Layer("bvlc_alexnet","conv3","conv",true,1,1));
    network.emplace_back(Layer("bvlc_alexnet","conv4","conv",true,1,1));
    network.emplace_back(Layer("bvlc_alexnet","conv5","conv",true,1,1));
    network.emplace_back(Layer("bvlc_alexnet","fc6","fc",true,1,0));
    network.emplace_back(Layer("bvlc_alexnet","fc7","fc",true,1,0));
    network.emplace_back(Layer("bvlc_alexnet","fc8","fc",false,1,0));
    return network;
}

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

// SCNN functions

void computePE(int n, int W, int H, int K, int stride, const float* act_queue, const int* act_queue_x,
        const int* act_queue_y, uint64_t act_queue_size, const float* wgt_queue, const int* wgt_queue_k,
        const int* wgt_queue_r, const int* wgt_queue_s, uint64_t wgt_queue_size, float* output_activations) {

    for(uint64_t i = 0; i < act_queue_size; i+=I) {
        for(uint64_t f = 0; f < wgt_queue_size; f+=F) {

            for(uint64_t ii = i; ii < std::min(i + I, act_queue_size); ii++) {
                for(uint64_t ff = f; ff < std::min(f + F, wgt_queue_size); ff++) {

                    auto act = act_queue[ii];
                    auto x = act_queue_x[ii];
                    auto y = act_queue_y[ii];

                    auto wgt = wgt_queue[ff];
                    auto k = wgt_queue_k[ff];
                    auto r = wgt_queue_r[ff];
                    auto s = wgt_queue_s[ff];

                    int w = (x - r) / stride;
                    int h = (y - s) / stride;

                    if(w >= 0 && w < W && h >= 0 && h < H) {
                        auto pos = n * W * H * K + k * W * H + w * H + h;
                        output_activations[pos] += act * wgt;
                    }

                }
            }

        }
    }
}

void computeTile(int n, int ct, int ck, int kc, int tw, int th, int Kc, int X, int Y, int K, int W, int H, int R, int S,
        const Layer &layer, float* output_activations) {

    int padding = layer.padding;
    int stride = layer.stride;

    // Iterate PEs
    for(int pex = 0; pex < Wt; pex++) {
        for(int pey = 0; pey < Ht; pey++) {

            int x_begin = pex * tw, y_begin = pey * th, k_begin = kc;
            int x_end = std::min(x_begin + tw, X), y_end = std::min(y_begin + th, Y), k_end = k_begin + Kc;

            // Iterate strides
            for(int sx = 0; sx < stride; sx++) {
                for(int sy = 0; sy < stride; sy++) {

                    // Count number of effectual activations
                    uint64_t act_queue_count = 0;
                    for(int x = x_begin; x < x_end; x++) {
                        int tmp_sx = x % stride;
                        for(int y = y_begin; y < y_end; y++) {
                            int tmp_sy = y % stride;
                            auto act_bits = layer.act_get(n,ct+ck,x,y);
                            if(act_bits != 0 && sx == tmp_sx && sy == tmp_sy) act_queue_count++;
                        }
                    }

                    // Count number of effectual weights
                    uint64_t wgt_queue_count = 0;
                    for(int r = 0; r < R; r++) {
                        int tmp_sx = (r + padding) % stride;
                        for(int s = 0; s < S; s++) {
                            int tmp_sy = (s + padding) % stride;
                            for(int k = k_begin; k < k_end; k++) {
                                auto wgt_bits = layer.wgt_get(k,ck,r,s);
                                if(wgt_bits != 0 && sx == tmp_sx && sy == tmp_sy) wgt_queue_count++;
                            }
                        }
                    }

                    // Allocate space for the queues
                    auto act_queue = (float *) malloc(act_queue_count * sizeof(float));
                    if (act_queue == nullptr) {
                        fprintf(stderr, "Error: Failed to allocate activations queue!\n");
                        exit(EXIT_FAILURE);
                    }
                    auto act_queue_x = ((int *) malloc(act_queue_count * sizeof(int)));
                    if (act_queue_x == nullptr) {
                        fprintf(stderr, "Error: Failed to allocate activations queue x!\n");
                        exit(EXIT_FAILURE);
                    }
                    auto act_queue_y = ((int *) malloc(act_queue_count * sizeof(int)));
                    if (act_queue_y == nullptr) {
                        fprintf(stderr, "Error: Failed to allocate activations queue y!\n");
                        exit(EXIT_FAILURE);
                    }

                    auto wgt_queue = (float *) malloc(wgt_queue_count * sizeof(float));
                    if (wgt_queue == nullptr) {
                        fprintf(stderr, "Error: Failed to allocate weights queue!\n");
                        exit(EXIT_FAILURE);
                    }
                    auto wgt_queue_k = ((int *) malloc(wgt_queue_count * sizeof(int)));
                    if (wgt_queue_k == nullptr) {
                        fprintf(stderr, "Error: Failed to allocate weights queue k!\n");
                        exit(EXIT_FAILURE);
                    }
                    auto wgt_queue_r = ((int *) malloc(wgt_queue_count * sizeof(int)));
                    if (wgt_queue_r == nullptr) {
                        fprintf(stderr, "Error: Failed to allocate weights queue r!\n");
                        exit(EXIT_FAILURE);
                    }
                    auto wgt_queue_s = ((int *) malloc(wgt_queue_count * sizeof(int)));
                    if (wgt_queue_s == nullptr) {
                        fprintf(stderr, "Error: Failed to allocate weights queue s!\n");
                        exit(EXIT_FAILURE);
                    }

                    // Populate activations queue
                    uint64_t index = 0;
                    for(int x = x_begin; x < x_end; x++) {
                        int tmp_sx = x % stride;
                        for(int y = y_begin; y < y_end; y++) {
                            int tmp_sy = y % stride;
                            auto act_bits = layer.act_get(n,ct+ck,x,y);
                            if(act_bits != 0 && sx == tmp_sx && sy == tmp_sy) {
                                act_queue[index] = act_bits;
                                act_queue_x[index] = x;
                                act_queue_y[index] = y;
                                index++;
                            }
                        }
                    }

                    // Populate weights queue
                    index = 0;
                    for(int r = 0; r < R; r++) {
                        int tmp_sx = (r + padding) % stride;
                        for(int s = 0; s < S; s++) {
                            int tmp_sy = (s + padding) % stride;
                            for(int k = k_begin; k < k_end; k++) {
                                auto wgt_bits =layer.wgt_get(k,ck,r,s);
                                if (wgt_bits != 0 && sx == tmp_sx && sy == tmp_sy) {
                                    wgt_queue[index] = wgt_bits;
                                    wgt_queue_k[index] = k;
                                    wgt_queue_r[index] = r;
                                    wgt_queue_s[index] = s;
                                    index++;
                                }
                            }
                        }
                    }

                    computePE(n,W,H,K,stride,act_queue,act_queue_x,act_queue_y,act_queue_count,wgt_queue, wgt_queue_k,
                            wgt_queue_r,wgt_queue_s, wgt_queue_count, output_activations);

                    free(act_queue);
                    free(act_queue_x);
                    free(act_queue_y);

                    free(wgt_queue);
                    free(wgt_queue_k);
                    free(wgt_queue_r);
                    free(wgt_queue_s);

                }
            }

        }
    }

}


// MAIN

int main(int argc, char *argv[]) {

    auto network = read_bvlc_alexnet();

    for(auto layer : network) {

        read_layer(layer);

        if(layer.type == "fc") {
            layer.reshape_to_2D();
            auto C = layer.act_shape[1];
            layer.act_split_4D((unsigned)(C / 256), 16, 16);

            auto Ck = layer.wgt_shape[1];
            layer.wgt_split_4D((unsigned)(Ck / 256), 16, 16);
        }

        layer.zero_pad();
        auto N = (int) layer.act_shape[0];
        auto C = (int) layer.act_shape[1];
        auto X = (int) layer.act_shape[2];
        auto Y = (int) layer.act_shape[3];

        auto K = (int) layer.wgt_shape[0];
        auto Ck = (int) layer.wgt_shape[1];
        auto R = (int) layer.wgt_shape[2];
        auto S = (int) layer.wgt_shape[3];

        int stride = layer.stride;

        int W = (X - R)/stride + 1;
        int H = (Y - S)/stride + 1;

        int groups = C / Ck;
        int Kc = K / groups;
        int kc = 0;

        X = (int)(ceil(X/(double)Wt))*Wt;
        Y = (int)(ceil(Y/(double)Ht))*Ht;
        auto tw = X/Wt;
        auto th = Y/Wt;

        layer.grid_zero_pad(X ,Y);

        // Initialize variables
        std::vector<size_t> output_shape;
        output_shape.push_back((unsigned) N);
        output_shape.push_back((unsigned) K);
        output_shape.push_back((unsigned) W);
        output_shape.push_back((unsigned) H);

        auto output_activations = (float *) malloc(N * K * W * H * sizeof(float));
        if (output_activations == nullptr) {
            fprintf(stderr, "Error: Failed to allocate output activations!\n");
            exit(EXIT_FAILURE);
        }

        // Add biases
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                for (int w = 0; w < W; w++) {
                    for (int h = 0; h < H; h++) {
                        auto pos = n * W * H * K + k * W * H + w * H + h;
                        output_activations[pos] = layer.bias[k];
                    }
                }
            }
        }

        for(int n = 0; n < N; n++) {
            for(int ct = 0; ct < C; ct+=Ck) {
                for(int ck = 0; ck < Ck; ck++) {
                    computeTile(n,ct,ck,kc,tw,th,Kc,X,Y,K,W,H,R,S,layer,output_activations);
                }
                kc += Kc;
            }
        }

        if (layer.ReLU) {
            for(uint64_t i = 0; i < (N * K * W * H); i++)
                output_activations[i] = ReLU(output_activations[i]);
        }

        check_values(layer,output_activations);
        free(output_activations);

    }

    return 0;
}

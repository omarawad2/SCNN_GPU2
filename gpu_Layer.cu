#include "Layer.h"

Layer::Layer(const std::string &_network, const std::string &_name, const std::string &_type, bool _ReLU, int _stride,
        int _padding) : ReLU(_ReLU), stride(_stride), padding(_padding) {
    this->network = _network;
    this->name = _name;
    this->type = _type;
	this->init = false;
}

Layer::~Layer() {
	if(init) {
		cudaFreeHost(weights);
		cudaFreeHost(bias);
		cudaFreeHost(activations);
		cudaFreeHost(output_activations);
	}
}

float Layer::act_get(int i, int j, int k, int l) const {
    uint32_t index = act_shape[1]*act_shape[2]*act_shape[3]*i + act_shape[2]*act_shape[3]*j + act_shape[3]*k + l;
    return activations[index];
}

float Layer::wgt_get(int i, int j, int k, int l) const {
    uint32_t index = wgt_shape[1]*wgt_shape[2]*wgt_shape[3]*i + wgt_shape[2]*wgt_shape[3]*j + wgt_shape[3]*k + l;
    return weights[index];
}

uint64_t getMaxIndex(const std::string &array) const {
    if(array == "weights") {
        return wgt_shape[0]*wgt_shape[1]*wgt_shape[2]*wgt_shape[3];
    } else if(array == "bias") {
        return bias_shape[0];
    } else if(array == "activations") {
        #ifdef FORCE_ONE_IMAGE
        return 1*act_shape[1]*act_shape[2]*act_shape[3];
        #else
        return act_shape[0]*act_shape[1]*act_shape[2]*act_shape[3];
        #endif
    } else if(array == "output_activations") {
        #ifdef FORCE_ONE_IMAGE
        if(out_act_shape.size() == 4) return 1*out_act_shape[1]*out_act_shape[2]*out_act_shape[3];
        else return 1*out_act_shape[1];
        #else
        if(out_act_shape.size() == 4) return out_act_shape[0]*out_act_shape[1]*out_act_shape[2]*out_act_shape[3];
        else return out_act_shape[0]*out_act_shape[1];
        #endif
    } else return 0;
}

void Layer::zero_pad() {

    #ifdef FORCE_ONE_IMAGE
    auto batch_size = (unsigned)1;
    #else
    auto batch_size = act_shape[0];
    #endif
    int act_channels = act_shape[1];
    int Nx = act_shape[2];
    int Ny = act_shape[3];
    int new_Nx = Nx + 2*padding;
    int new_Ny = Ny + 2*padding;

    uint64_t new_max_index = batch_size * act_channels * new_Nx * new_Ny;
    float* tmp_activations;
    cudaMallocHost((void **) &tmp_activations, new_max_index * sizeof(float));
    if (tmp_activations == NULL) {
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
                    uint32_t index_out = act_channels*new_Nx*new_Ny*n + new_Nx*new_Ny*k + new_Ny*(padding + i) +
                            (padding + j);
                    uint32_t index_in = act_channels*Nx*Ny*n + Nx*Ny*k + Ny*i + j;
                    tmp_activations[index_out] = activations[index_in];
                }
            }
        }
    }

    cudaFreeHost(activations);
    activations = tmp_activations;
    act_shape.clear();
    act_shape.push_back(batch_size);
    act_shape.push_back(act_channels);
    act_shape.push_back(new_Nx);
    act_shape.push_back(new_Ny);

}

void Layer::act_split_4D(int K, int X, int Y) {

    #ifdef FORCE_ONE_IMAGE
    auto batch_size = (unsigned)1;
    #else
    auto batch_size = act_shape[0];
    #endif
    int act_channels = act_shape[1];
    int Nx = act_shape[2];
    int Ny = act_shape[3];

    uint64_t new_max_index = batch_size * K * X * Y;
    float* tmp_activations;
    cudaMallocHost((void **) &tmp_activations, new_max_index * sizeof(float));
    if (tmp_activations == NULL) {
        fprintf(stderr, "Error: Failed to allocate padded activations!\n");
        exit(EXIT_FAILURE);
    }

    for(int n = 0; n < batch_size; n++) {
        for (int k = 0; k < act_channels; k++) {
            for (int i = 0; i < Nx; i++) {
                for(int j = 0; j < Ny; j++) {
                    int new_k = k / (X*Y);
                    int rem = k % (X*Y);
                    int new_i = rem / Y;
                    int new_j = rem % Y;
                    uint32_t index_out = K*X*Y*n + X*Y*new_k + Y*new_i + new_j;
                    uint32_t index_in = act_channels*Nx*Ny*n + Nx*Ny*k + Ny*i + j;
                    tmp_activations[index_out] = activations[index_in];
                }
            }
        }
    }

    cudaFreeHost(activations);
    activations = tmp_activations;
    act_shape.clear();
    act_shape.push_back(batch_size);
    act_shape.push_back((unsigned)K);
    act_shape.push_back((unsigned)X);
    act_shape.push_back((unsigned)Y);

}

void Layer::wgt_split_4D(int K, int X, int Y) {

    int num_filters = wgt_shape[0];
    int wgt_channels = wgt_shape[1];
    int Kx = wgt_shape[2];
    int Ky = wgt_shape[3];

    uint64_t new_max_index = num_filters * K * X * Y;
    float* tmp_weights;
    cudaMallocHost((void **) &tmp_weights, new_max_index * sizeof(float));
    if (tmp_weights == NULL) {
        fprintf(stderr, "Error: Failed to allocate padded weights!\n");
        exit(EXIT_FAILURE);
    }

    for(int n = 0; n < num_filters; n++) {
        for (int k = 0; k < wgt_channels; k++) {
            for (int i = 0; i < Kx; i++) {
                for(int j = 0; j < Ky; j++) {
                    int new_k = k / (X*Y);
                    int rem = k % (X*Y);
                    int new_i = rem / Y;
                    int new_j = rem % Y;
                    uint32_t index_out = K*X*Y*n + X*Y*new_k + Y*new_i + new_j;
                    uint32_t index_in = wgt_channels*Kx*Ky*n + Kx*Ky*k + Ky*i + j;
                    tmp_weights[index_out] = weights[index_in];
                }
            }
        }
    }

    cudaFreeHost(weights);
    weights = tmp_weights;
    wgt_shape.clear();
    wgt_shape.push_back(num_filters);
    wgt_shape.push_back((unsigned)K);
    wgt_shape.push_back((unsigned)X);
    wgt_shape.push_back((unsigned)Y);

}

void Layer::reshape_to_2D() {

    #ifdef FORCE_ONE_IMAGE
    auto batch_size = (unsigned)1;
    #else
    auto batch_size = act_shape[0];
    #endif
    int act_channels = act_shape[1];
    int Nx = act_shape[2];
    int Ny = act_shape[3];
    int new_act_channels = act_channels * Nx * Ny;

    act_shape.clear();
    act_shape.push_back(batch_size);
    act_shape.push_back(new_act_channels);
    act_shape.push_back(1);
    act_shape.push_back(1);

}

inline
cudaError_t check_error(cudaError_t err, std::string task) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: Failed to %s (error code: %s)!\n", task.c_str(), cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return err;
}

// Read network from numpy arrays
void Layer::read_layer() {

    cnpy::NpyArray data_npy;
    uint64_t max_index;

    cnpy::npy_load("net_traces/" + network + "/wgt-" + name + ".npy" , data_npy, wgt_shape);
    max_index = getMaxIndex("weights");
    check_error(cudaMallocHost((void **) &weights, max_index * sizeof(float)),"allocate layer weights");

    for(uint32_t i = 0; i < max_index; i++)
        weights[i] = data_npy.data<float>()[i];

    cnpy::npy_load("net_traces/" + network + "/bias-" + name + ".npy" , data_npy, bias_shape);
    max_index = getMaxIndex("bias");
    check_error(cudaMallocHost((void **) &bias, max_index * sizeof(float)),"allocate layer bias");

    for(uint32_t i = 0; i < max_index; i++)
        bias[i] = data_npy.data<float>()[i];

    cnpy::npy_load("net_traces/" + network + "/act-" + name + "-0.npy" , data_npy, act_shape);
    max_index = getMaxIndex("activations");
    check_error(cudaMallocHost((void **) &activations, max_index * sizeof(float)),"allocate layer activations");

    for(uint32_t i = 0; i < max_index; i++)
        activations[i] = data_npy.data<float>()[i];

    cnpy::npy_load("net_traces/" + network + "/act-" + name + "-0-out.npy" , data_npy, out_act_shape);
    max_index = getMaxIndex("output_activations");
    check_error(cudaMallocHost((void **) &output_activations, max_index * sizeof(float)),"allocate layer output activations");

    for(uint32_t i = 0; i < max_index; i++)
        output_activations[i] = data_npy.data<float>()[i];

	this->init = true;
	
	#ifdef VERBOSE
    printf("Layer %s loaded into memory\n",name.c_str());
	#endif

}

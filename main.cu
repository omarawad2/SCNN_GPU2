
#include "Layer.h"

#define GLOBAL_TIME

//############################################### Read networks ########################################################

std::vector<Layer> read_bvlc_alexnet() {
    std::vector<Layer> network;
    network.push_back(Layer("bvlc_alexnet","conv1","conv",true,4,0));
    network.push_back(Layer("bvlc_alexnet","conv2","conv",true,1,2));
    network.push_back(Layer("bvlc_alexnet","conv3","conv",true,1,1));
    network.push_back(Layer("bvlc_alexnet","conv4","conv",true,1,1));
    network.push_back(Layer("bvlc_alexnet","conv5","conv",true,1,1));
    network.push_back(Layer("bvlc_alexnet","fc6","fc",true,1,0));
    network.push_back(Layer("bvlc_alexnet","fc7","fc",true,1,0));
    network.push_back(Layer("bvlc_alexnet","fc8","fc",false,1,0));
    return network;
}

//############################################### Auxiliary functions ##################################################

inline
cudaError_t check_error(cudaError_t err, std::string task) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: Failed to %s (error code: %s)!\n", task.c_str(), cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return err;
}


template <typename T>
void print(int size, const T a){
    for(int i=0; i<size;i++)
        printf("%4.4f;",a[i]);
    printf("\n");
}

template <typename T>
T* host2Dev(uint64_t size, const T *h_data, std::string task){
    T* d_data;
    check_error(cudaMalloc((void**) &d_data, size*sizeof(T)),task);
    check_error(cudaMemcpy(d_data, h_data, size*sizeof(T), cudaMemcpyHostToDevice),task);

    return d_data;
}

// Checking function
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

//############################################### CUDA SCNN ############################################################

//naive implementation
__global__ void kAddBias(int N, int K, int W, int H, const float* d_bias, float* d_output_activations){

    int n = threadIdx.x + blockIdx.x*blockDim.x;
    int k = threadIdx.y + blockIdx.y*blockDim.y;

    //TODO: try different configurations
    if(n < N && k < K){
        for(int w =0; w < W; w++){
            for(int h=0; h<H; h++){
                int pos = n * W * H * K + k * W * H + w * H + h;
                d_output_activations[pos] = d_bias[k];
            }
        }
    }
}

__global__ void kRelu(int N, int K, int W, int H, float* d_output_activations){
    
    int x = threadIdx.x + blockIdx.x*blockDim.x;

    if(x < N*K*W*H){
    	// Maybe a max with 0 would be faster?
        d_output_activations[x] = (d_output_activations[x] > 0) ? d_output_activations[x] : 0;
        //d_output_activations[x] = std::max(d_output_activations[x],0);
    }
}

//naive implmentation
__global__ void kComputePE(int n, int W, int H, int K, int stride, const float* d_act_queue, const int* d_act_queue_x, 
    const int* d_act_queue_y, uint64_t act_queue_size, const float* d_wgt_queue, const int* d_wgt_queue_k,
    const int* d_wgt_queue_r, const int* d_wgt_queue_s, uint64_t wgt_queue_size, float* d_output_activations){
    //TODO: use shared mem.
    //TODO: try different configurations

    int ii = threadIdx.x + blockIdx.x*blockDim.x;
    //int ff = threadIdx.y + blockIdx.y*blockDim.y;

    if(ii < act_queue_size){
        float act = d_act_queue[ii];
        int x = d_act_queue_x[ii];
        int y = d_act_queue_y[ii];
        for(int ff = 0; ff < wgt_queue_size; ff++){ //TODO usually weight queue is way bigger, may me better to parallelize this
            float wgt = d_wgt_queue[ff];
            int k = d_wgt_queue_k[ff];
            int r = d_wgt_queue_r[ff];
            int s = d_wgt_queue_s[ff];

            //TODO: try to remove div. (takes a lot on GPU)
            int w = (x-r)/stride;
            int h = (y-s)/stride;

             if(w >= 0 && w < W && h >= 0 && h < H) {
                int pos = n * W * H * K + k * W * H + w * H + h;
                //TODO: memory access not coalesced
                d_output_activations[pos] += act * wgt;
            }
        }
    }
}

//naive implementation
__global__ void kCount_effectual_activations(int n, int channel, int sx, int sy, int C, int X, int Y, int stride,
        const float* d_act, int *d_act_queue_count) {

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x < X){
        int tmp_sx = x % stride;
        if(y < Y){
            int tmp_sy = y % stride;
            int pos = C*X*Y*n + X*Y*channel + X*x + y;
            float act_bits = d_act[pos];
            if(act_bits !=0 && sx == tmp_sx && sy == tmp_sy){
                atomicAdd(d_act_queue_count,1);
            }
        }
    }
}

//naive implementation
__global__ void kPopulate_effectual_activations(int n, int channel, int sx, int sy, int C, int X, int Y, int stride,
        const float* d_act,float* d_act_queue, int* d_act_queue_x, int* d_act_queue_y) {

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x < X){
        int tmp_sx = x % stride;
        if(y < Y){
            int tmp_sy = y % stride;
            int pos = C*X*Y*n + X*Y*channel + X*x + y;
            float act_bits = d_act[pos];
            if(act_bits !=0 && sx == tmp_sx && sy == tmp_sy){
                int index = x+y*X;
                d_act_queue[index] = act_bits;
                d_act_queue_x[index] = x;
                d_act_queue_y[index] = y;
            }
        }
    }
}

//naive implementation
__global__ void kCount_effectual_weights(int ck, int sx, int sy,int k_begin, int k_end, int Ck, int R, int S,
        int stride, int padding, const float* d_wgt, int *d_wgt_queue_count){
    
    int r = threadIdx.x + blockIdx.x*blockDim.x;
    int s = threadIdx.y + blockIdx.y*blockDim.y;
    int k = threadIdx.z + blockIdx.z*blockDim.z;

    if(r < R){
        int tmp_sx = (r + padding) % stride;
        if(s < S){
            int tmp_sy = (s + padding) % stride;
            if(k < k_end){
                int pos = Ck*R*S*(k_begin + k) + R*S*ck + S*r + r;
                float wgt_bits = d_wgt[pos];
                if(wgt_bits != 0 && sx == tmp_sx && sy == tmp_sy){
                    atomicAdd(d_wgt_queue_count,1);
                }
            }
        }
    }
}

//naive implementation
__global__ void kPopulate_effectual_weights(int ck, int sx, int sy,int k_begin, int k_end, int Ck, int R, int S,
        int stride, int padding, const float* d_wgt, float* d_wgt_queue, int* d_wgt_queue_k, int* d_wgt_queue_r,
        int* d_wgt_queue_s){

    int r = threadIdx.x + blockIdx.x*blockDim.x;
    int s = threadIdx.y + blockIdx.y*blockDim.y;
    int k = threadIdx.z + blockIdx.z*blockDim.z;

    if(r < R){
        int tmp_sx = (r + padding) % stride;
        if(s < S){
            int tmp_sy = (s + padding) % stride;
            if(k < k_end){
                int pos = Ck*R*S*(k_begin + k) + R*S*ck + S*r + r;
                float wgt_bits = d_wgt[pos];
                if(wgt_bits != 0 && sx == tmp_sx && sy == tmp_sy){
                    int index = r+s*R+k*R*S;
                    d_wgt_queue[index] = wgt_bits;
                    d_wgt_queue_k[index] = k;
                    d_wgt_queue_r[index] = r;
                    d_wgt_queue_s[index] = s;
                }
            }
        }
    }
}

//############################################### CPU SCNN #############################################################

void addBias(int N, int K, int W, int H, const Layer &layer, float* d_output_activations) {

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #endif

	float* d_bias = host2Dev(layer.getMaxIndex("bias"), layer.bias,"allocate device bias");

    dim3 block(32, 32);
    dim3 grid((N+block.x-1)/block.x,(K+block.y-1)/block.y);
    kAddBias<<<grid, block>>>(N,K,W,H,d_bias,d_output_activations);
    cudaDeviceSynchronize();

    check_error(cudaFree(d_bias),"free device bias");

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    printf("kAddBias block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kAddBias time %.6f\n",time_span.count());
    #endif
}

void relu(int N, int K, int W, int H, const Layer &layer, float* d_output_activations) {

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #endif
    
    if(layer.ReLU){
        dim3 block(1024, 1);
        dim3 grid((N*K*W*H+block.x-1)/block.x,1);
        kRelu<<<grid,block>>>(N,K,W,H,d_output_activations);
        cudaDeviceSynchronize();
    }

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    printf("kRelu block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kRelu time %.6f\n",time_span.count());
    #endif
}

float* count_effectual_activations(int n, int channel, int sx, int sy, int stride, const Layer &layer,
        int &h_act_queue_count) {

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #endif

    //int N = 1; // Force one image, (int) layer.act_shape[0];
    int C = (int) layer.act_shape[1];
    int X = (int) layer.act_shape[2];
    int Y = (int) layer.act_shape[3];

    // TODO allocate only one channel
    float *d_act = host2Dev(layer.getMaxIndex("activations"), layer.activations,"allocate device activations channel");
    int *d_act_queue_count = host2Dev(1, &h_act_queue_count,"allocate device activations counter");

    dim3 block(32, 32);
    dim3 grid((X+block.x-1)/block.x,(Y+block.y-1)/block.y);
    //TODO:add streams
    kCount_effectual_activations<<<grid,block>>>(n,channel,sx,sy,C,X,Y,stride,d_act,d_act_queue_count);
    cudaDeviceSynchronize();

    check_error(cudaMemcpy(&h_act_queue_count, d_act_queue_count, 4, cudaMemcpyDeviceToHost),
            "copy activations counter from device to host");

    check_error(cudaFree(d_act_queue_count),"free device activations counter");

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    printf("kCount_effectual_activations block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kCount_effectual_activations time %.6f\n",time_span.count());
    #endif

    return d_act;

}

void populate_effectual_activations(int n, int channel, int sx, int sy, int stride, const Layer &layer, float *d_act,
        float *d_act_queue, int *d_act_queue_x, int *d_act_queue_y) {

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #endif

    //int N = 1; // Force one image, (int) layer.act_shape[0];
    int C = (int) layer.act_shape[1];
    int X = (int) layer.act_shape[2];
    int Y = (int) layer.act_shape[3];

    dim3 block(32, 32);
    dim3 grid((X+block.x-1)/block.x,(Y+block.y-1)/block.y);
    //TODO:add streams
    kPopulate_effectual_activations<<<grid,block>>>(n,channel,sx,sy,C,X,Y,stride,d_act,d_act_queue,d_act_queue_x,
            d_act_queue_y);
    cudaDeviceSynchronize();

    check_error(cudaFree(d_act),"free device activations channel");

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    printf("kPopulate_effectual_activations block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kPopulate_effectual_activations time %.6f\n",time_span.count());
    #endif

}

float* count_effectual_weights(int ck, int sx, int sy, int k_begin, int k_end, int stride, int padding,
        const Layer &layer, int &h_wgt_queue_count) {

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #endif

    //int K = (int) layer.wgt_shape[0];
    int Ck = (int) layer.wgt_shape[1];
    int R = (int) layer.wgt_shape[2];
    int S = (int) layer.wgt_shape[3];

    // TODO allocate only one channel and only set of working weighs (two towers alexnet)
    float* d_wgt = host2Dev(layer.getMaxIndex("weights"), layer.weights,"allocate device weights channel");
    int *d_wgt_queue_count = host2Dev(1, &h_wgt_queue_count,"allocate device weights counter");

    dim3 block(16, 16, 4);
    dim3 grid((R+block.x-1)/block.x,(S+block.y-1)/block.y,(k_end+block.z-1)/block.z);
    //TODO:add streams
    kCount_effectual_weights<<<grid,block>>>(ck,sx,sy,k_begin,k_end,Ck,R,S,stride,padding,d_wgt,d_wgt_queue_count);
    cudaDeviceSynchronize();

    check_error(cudaMemcpy(&h_wgt_queue_count, d_wgt_queue_count, 4, cudaMemcpyDeviceToHost),
                "copy weights counter from device to host");

    check_error(cudaFree(d_wgt_queue_count),"free device weights counter");

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    printf("kCount_effectual_weights block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kCount_effectual_weights time %.6f\n",time_span.count());
    #endif

    return d_wgt;

}

void populate_effectual_weights(int ck, int sx, int sy, int k_begin, int k_end, int stride, int padding,
        const Layer &layer, float *d_wgt, float* d_wgt_queue, int* d_wgt_queue_k, int* d_wgt_queue_r,
        int* d_wgt_queue_s) {

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #endif

    //int K = (int) layer.wgt_shape[0];
    int Ck = (int) layer.wgt_shape[1];
    int R = (int) layer.wgt_shape[2];
    int S = (int) layer.wgt_shape[3];

    dim3 block(16, 16, 4);
    dim3 grid((R+block.x-1)/block.x,(S+block.y-1)/block.y,(k_end+block.z-1)/block.z);
    //TODO:add streams
    kPopulate_effectual_weights<<<grid,block>>>(ck,sx,sy,k_begin,k_end,Ck,R,S,stride,padding,d_wgt,d_wgt_queue,
            d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s);
    cudaDeviceSynchronize();

    check_error(cudaFree(d_wgt),"free device weights channel");

    #ifndef GLOBAL_TIME
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    printf("kPopulate_effectual_weights block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kPopulate_effectual_weights time %.6f\n",time_span.count());
    #endif

}

void computePE(int n, int W, int H, int K, int stride, const float* act_queue, const int* act_queue_x,
               const int* act_queue_y, uint64_t act_queue_size, const float* wgt_queue, const int* wgt_queue_k,
               const int* wgt_queue_r, const int* wgt_queue_s, uint64_t wgt_queue_size, float* output_activations) {

    for(uint64_t i = 0; i < act_queue_size; i+=4) {
        for(uint64_t f = 0; f < wgt_queue_size; f+=4) {

            for(uint64_t ii = i; ii < std::min(i + 4, act_queue_size); ii++) {
                for(uint64_t ff = f; ff < std::min(f + 4, wgt_queue_size); ff++) {

                    float act = act_queue[ii];
                    int x = act_queue_x[ii];
                    int y = act_queue_y[ii];

                    float wgt = wgt_queue[ff];
                    int k = wgt_queue_k[ff];
                    int r = wgt_queue_r[ff];
                    int s = wgt_queue_s[ff];

                    int w = (x - r) / stride;
                    int h = (y - s) / stride;

                    if(w >= 0 && w < W && h >= 0 && h < H) {
                        int pos = n * W * H * K + k * W * H + w * H + h;

                        #pragma omp atomic
                        output_activations[pos] += act * wgt;
                    }

                }
            }

        }
    }
}

void computeTile(int n, int ct, int ck, int kc, int Kc, int K, int W, int H, const Layer &layer,
        float* d_output_activations) {

    int padding = layer.padding;
    int stride = layer.stride;

    // Iterate PEs

    int k_begin = kc;
    int k_end = k_begin + Kc;

    // Iterate strides
    for(int sx = 0; sx < stride; sx++) {
        for(int sy = 0; sy < stride; sy++) {

            // Count number of effectual activations
            int act_queue_count = 0;
            float *d_act = count_effectual_activations(n,ct+ck,sx,sy,stride,layer,act_queue_count);

            // Count number of effectual weights
            int wgt_queue_count = 0;
            float *d_wgt = count_effectual_weights(ck,sx,sy,k_begin,k_end,stride,padding,layer,wgt_queue_count);

            float *d_act_queue, *d_wgt_queue;
            int *d_act_queue_x, *d_act_queue_y;
            int *d_wgt_queue_k, *d_wgt_queue_r, *d_wgt_queue_s;

            // Allocate space for the queues on device
            check_error(cudaMalloc((void**) &d_act_queue, act_queue_count*sizeof(float)),"allocate device activations queue");
            check_error(cudaMalloc((void**) &d_act_queue_x, act_queue_count*sizeof(int)),"allocate device activations queue X dim");
            check_error(cudaMalloc((void**) &d_act_queue_y, act_queue_count*sizeof(int)),"allocate device activations queue Y dim");
            check_error(cudaMalloc((void**) &d_wgt_queue, wgt_queue_count*sizeof(float)),"allocate device weights queue");
            check_error(cudaMalloc((void**) &d_wgt_queue_k, wgt_queue_count*sizeof(int)),"allocate device weights queue K filter");
            check_error(cudaMalloc((void**) &d_wgt_queue_r, wgt_queue_count*sizeof(int)),"allocate device weights queue R dim");
            check_error(cudaMalloc((void**) &d_wgt_queue_s, wgt_queue_count*sizeof(int)),"allocate device weights queue S dim");

            //FIX both populates
            // Populate activations queue
            populate_effectual_activations(n,ct+ck,sx,sy,stride,layer,d_act,d_act_queue,d_act_queue_x,d_act_queue_y);

            // Populate weights queue
            populate_effectual_weights(ck,sx,sy,k_begin,k_end,stride,padding,layer,d_wgt,d_wgt_queue,d_wgt_queue_k,
                    d_wgt_queue_r,d_wgt_queue_s);

            // ###############################Remove by GPU code {

            // Allocate space for the queues
            float *act_queue = (float*) malloc(act_queue_count * sizeof(float));
            int *act_queue_x = ((int*) malloc(act_queue_count * sizeof(int)));
            int *act_queue_y = ((int*) malloc(act_queue_count * sizeof(int)));
            float *wgt_queue = (float*) malloc(wgt_queue_count * sizeof(float));
            int *wgt_queue_k = ((int*) malloc(wgt_queue_count * sizeof(int)));
            int *wgt_queue_r = ((int*) malloc(wgt_queue_count * sizeof(int)));
            int *wgt_queue_s = ((int*) malloc(wgt_queue_count * sizeof(int)));

            check_error(cudaMemcpy(act_queue, d_act_queue, act_queue_count*sizeof(float), cudaMemcpyDeviceToHost),"");
            check_error(cudaMemcpy(act_queue_x, d_act_queue_x, act_queue_count*sizeof(int), cudaMemcpyDeviceToHost),"");
            check_error(cudaMemcpy(act_queue_y, d_act_queue_y, act_queue_count*sizeof(int), cudaMemcpyDeviceToHost),"");
            check_error(cudaMemcpy(wgt_queue, d_wgt_queue, wgt_queue_count*sizeof(float), cudaMemcpyDeviceToHost),"");
            check_error(cudaMemcpy(wgt_queue_k, d_wgt_queue_k, wgt_queue_count*sizeof(int), cudaMemcpyDeviceToHost),"");
            check_error(cudaMemcpy(wgt_queue_r, d_wgt_queue_r, wgt_queue_count*sizeof(int), cudaMemcpyDeviceToHost),"");
            check_error(cudaMemcpy(wgt_queue_s, d_wgt_queue_s, wgt_queue_count*sizeof(int), cudaMemcpyDeviceToHost),"");

            computePE(n,W,H,K,stride,act_queue,act_queue_x,act_queue_y,act_queue_count,wgt_queue, wgt_queue_k,
                      wgt_queue_r,wgt_queue_s, wgt_queue_count, d_output_activations);

            free(act_queue);
            free(act_queue_x);
            free(act_queue_y);

            free(wgt_queue);
            free(wgt_queue_k);
            free(wgt_queue_r);
            free(wgt_queue_s);

            // ###############################Remove by GPU code }

            //free GPU resources
            check_error(cudaFree(d_act_queue),"free device activations queue");
            check_error(cudaFree(d_act_queue_x),"free device activations queue X dim");
            check_error(cudaFree(d_act_queue_y),"free device activations queue Y dim");
            check_error(cudaFree(d_wgt_queue),"free device weights queue");
            check_error(cudaFree(d_wgt_queue_k),"free device weights queue K dim");
            check_error(cudaFree(d_wgt_queue_r),"free device weights queue R dim");
            check_error(cudaFree(d_wgt_queue_s),"free device weights queue S dim");

        }
    }

}

//############################################### Main #################################################################

int main(int argc, char *argv[]) {

    std::vector<Layer> network = read_bvlc_alexnet();

    for(int i = 0; i < network.size(); i++) {

	Layer layer = network[i];

        layer.read_layer();

        if(layer.type == "fc") {
            layer.reshape_to_2D();
            int C = layer.act_shape[1];
            layer.act_split_4D((unsigned)(C / 256), 16, 16);

            int Ck = layer.wgt_shape[1];
            layer.wgt_split_4D((unsigned)(Ck / 256), 16, 16);
        }

        layer.zero_pad();
        int N = 1; // Force one image, (int) layer.act_shape[0];
        int C = (int) layer.act_shape[1];
        int X = (int) layer.act_shape[2];
        int Y = (int) layer.act_shape[3];

        int K = (int) layer.wgt_shape[0];
        int Ck = (int) layer.wgt_shape[1];
        int R = (int) layer.wgt_shape[2];
        int S = (int) layer.wgt_shape[3];

        int stride = layer.stride;

        int W = (X - R)/stride + 1;
        int H = (Y - S)/stride + 1;

        int groups = C / Ck;
        int Kc = K / groups;
        int kc = 0;

        uint32_t bytes = N*K*W*H * sizeof(float);

        float* d_output_activations;
        check_error(cudaMalloc((void **) &d_output_activations, bytes),"allocate device output activations");

        addBias(N, K, W, H, layer, d_output_activations);

        // ############################## TEST ############################################ {
        float *h_output_activations = (float *) malloc(bytes);
        if (h_output_activations == NULL) {
            fprintf(stderr, "Error: Failed to allocate output activations!\n");
            exit(EXIT_FAILURE);
        }

        check_error(cudaMemcpy(h_output_activations, d_output_activations, bytes, cudaMemcpyDeviceToHost),
                    "copy output activations from device to host");
        // ############################## TEST ############################################ }

        //core compute
        for(int n = 0; n < N; n++) {
            for(int ct = 0; ct < C; ct+=Ck) {
                for(int ck = 0; ck < Ck; ck++) {
                    computeTile(n,ct,ck,kc,Kc,K,W,H,layer,h_output_activations);
                    //computeTile(n,ct,ck,kc,Kc,K,W,H,layer,d_output_activations);
                }
                kc += Kc;
            }
        }
        // ############################## TEST ############################################ {

        check_error(cudaMemcpy(d_output_activations, h_output_activations, bytes, cudaMemcpyHostToDevice),
                    "copy output activations from device to host");
        // ############################## TEST ############################################ }

        relu(N, K, W, H, layer, d_output_activations);

        /*float *h_output_activations = (float *) malloc(bytes);
        if (h_output_activations == NULL) {
            fprintf(stderr, "Error: Failed to allocate output activations!\n");
            exit(EXIT_FAILURE);
        }*/

        check_error(cudaMemcpy(h_output_activations, d_output_activations, bytes, cudaMemcpyDeviceToHost),
        		"copy output activations from device to host");

        check_values(layer,h_output_activations);
        free(h_output_activations);

    }

	return 0;
}

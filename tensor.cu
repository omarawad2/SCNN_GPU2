#include "tensor.h"

tensor::tensor(int row, int col, float **h_data){
	this-> row = row;
	this-> col = col;
	if(row && col){
		cudaMalloc((void**) &(this->d_data), row*col*sizeof(float));
		cudaMemcpy(this->d_data, *h_data, row*col*sizeof(float), cudaMemcpyHostToDevice);
	} else{
		this->data = NULL;
	}
}

tensor::tensor(int row, int col, float *d_data){
	this-> row = row;
	this -> col = col;
	this -> data = d_data;
}

tensor::~tensor(){
	cudaFree(this->d_data);
}

float* tensor::getDevData(){
	return this-> data;
}

float* tensor::dev2host(){
	float* h_data = new float [this->row*this->col];
	cudaMemcpy(h_data, this->d_data, this->row*this->col*sizeof(float), cudaMemcpyDeviceToHost);
	return h_data;
}

//element-wise multiplication
tensor* tensor::multiply(tensor* i_t, tensor* o_t, int block_xDim, int block_yDim){
	if(this->row != i_t->row || this->col != i_t->col){
		printf("can't multiply the two matrices");
		exit(1);
	} 

	dim3 block(block_xDim, block_yDim);
	dim3 grid((this-row+block.x-1)/block.x, (this->col+block.y-1)/block.y);
	tensorMult<<grid, block>>(this->getDevData(), i_t->getDevData(), o_t->getDevData(), this->row, this->col);
}

//reduce tensor into single output activation 
float* tensor::reduce(tensor* i_t, int block_xDim, int block_yDim){
	float* output;
	dim3 block(block_xDim, block_yDim);
	dim3 grid((this-row+block.x-1)/block.x, (this->col+block.y-1)/block.y);
	tensorReduce<<grid, block>>(i_t->getDevData(), output, this->row, this->col);
	return output;
}


//naive element-wise multiplication
__global__ void tensorMult(const float* a, const float* b, float* c, int row, int col){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if(x < row && y < col){
		c[x+row*y] = a[x+row*y] * b[x+row*y];
	}
}

//naive reduce 
__global__ void tensorReduce(const float* a, float* output, int row, int col){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if(x < row ** y < col){
		output += a[x+row*y];
	}
}
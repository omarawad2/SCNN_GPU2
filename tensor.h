#ifndef TENSOR_H
#define TENSOR_H

#include <cstdio>

struct tensor {

	int row;
	int col;
	float *data;

	tensor(int row, int col, float *d_data);
	tensor(int row, int col, float **h_data);
	~tensor();

	float* getDevData();
	float* dev2host();

	tensor* multiply(tensor* i_t, tensor* o_t, int block_xDim, int block_yDim);
	float* reduce(tensor* i_t, int block_xDim, int block_yDim);

};

#endif
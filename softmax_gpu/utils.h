#ifndef _UTILS_H_
#define _UTILS_H_

#include"error_check.h"
#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))

#define exp 2.718281828
#define BLOCK_SIZE 32

void weight_initialization(float* W, int W_rows, int W_columns)
{
	for(int i=0; i<W_rows*W_columns; i++)
	{
		W[i] = RAND_FLOAT();
	}
}

__global__ void kernel_matrix_transpose(float *input, float *output, int num_rows, int num_cols)
//用上改进版的shared mem的矩阵转置就报错，找不到问题在哪
{
	int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

	int input_width = num_cols;
	int output_width = num_rows;

	if(row_idx<num_rows && col_idx<num_cols)
	{
		output[col_idx*output_width+row_idx] = input[row_idx*input_width+col_idx];
	}
}

void matrix_transpose(float *input, float *output, int num_rows, int num_cols)
{
	dim3 blocks((num_cols-1)/BLOCK_SIZE+1, (num_rows-1)/BLOCK_SIZE+1);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	kernel_matrix_transpose<<<blocks, threads>>>(input, output,  num_rows,  num_cols);
    CHECK(cudaDeviceSynchronize());
	
}

__global__ void kernel_matrix_multiply(float *M, float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols)
//用上改进版的shared mem的矩阵乘法就报错，找不到问题在哪
{
	int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
    if(row<M_rows && col<N_cols)
    {
        float Pvalue = 0;
        for(int k=0; k<M_cols; k++){
            Pvalue += M[row*M_cols+k] * N[k*N_cols+col];
        }
        P[row*N_cols+col] = Pvalue;
    } 
}
void matrix_multiply( float *M,  float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols)
{

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N_cols-1)/block.y+1, (M_rows-1)/block.x+1);

    kernel_matrix_multiply<<<grid, block>>>(M, N, P, M_rows, M_cols, N_rows, N_cols);
	CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

__global__ void kernel_softmax(float* activations, int rows, int cols)
//本来想用shared对每个线程访问，直接存储每一列的最大值和他们的和，但好像一直出错，没找到问题在哪，就用了现在这个方法
{
	__shared__ float temp[10];
	int row =  threadIdx.x;

	int col = blockIdx.x;
	if(col<cols)
	{
		if(row<rows)
		{
			temp[row]=activations[row*cols+col];
		}
		__syncthreads();

		float max=temp[0];
		float sum=0;
		for(int i=1;i<rows;i++)
		{
			max=fmax(temp[i],max);
		}
		for(int i=0;i<rows;i++)
		{
			temp[i]=pow(exp,temp[i]-max);
			sum+=temp[i];
		}
		for(int i=0;i<rows;i++)
		{
			activations[i*cols+col]=temp[i]/sum;
		}
	}
}
void softmax(float* activations, int rows, int cols)
//对每一列进行softmax变化,每一个block是一列
{
    dim3 block(rows);
    dim3 grid(cols);

    kernel_softmax<<<cols, rows>>>(activations, rows, cols);
    CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

}

//下面两个函数感觉用cuda加速没太大必要，就没改了
float accuracy(float* activations, int* labels, int rows, int cols)
{
	float correct=0;
	for(int col_idx=0; col_idx<cols; col_idx++)
	{
		int max=0;
		int p_label=0;
		for(int row_idx=0; row_idx<rows; row_idx++)
		{
			if(activations[row_idx*cols+col_idx]>max)
			{
				max=activations[row_idx*cols+col_idx];
				p_label=row_idx;
			}
		}
		if(p_label==labels[col_idx])
			correct++;
	}
	return correct/(cols);
}


float cross_entropy_loss(float* activations, int* labels, int rows, int cols)
{
	float loss=0;
	for(int col_idx=0; col_idx<cols; col_idx++)
	{
		for(int row_idx=0; row_idx<rows; row_idx++)
		{
			if(row_idx==labels[col_idx])
			{
				if(activations[row_idx*cols+col_idx]==0)
					loss -=1 * log(activations[row_idx*cols+col_idx]+0.00001);
				else
					loss -=1 * log(activations[row_idx*cols+col_idx]);
			}
		}
	}
	return loss/(cols);
}

__global__ void kernel_delta_compute(float* delta, float* activations, int *labels, int W_rows, int number_of_samples)
//本来要梯度要除以样本数目，但除了之后好像（陷入局部最优，准确率一直徘徊在20-30），跳不出来了，就直接没除，改为更变学习率控制
{
	int x = blockIdx.x * blockDim.x ;
	int y = blockIdx.y * blockDim.y ;

	int col_idx = x + threadIdx.x;
	int row_idx = y + threadIdx.y;
	

	if(col_idx<number_of_samples)
	{
		if(row_idx<W_rows)
		{
			if(row_idx==labels[col_idx])
			{
				delta[row_idx*number_of_samples+col_idx] = activations[row_idx*number_of_samples+col_idx]-1;
			}
			else
				delta[row_idx*number_of_samples+col_idx] = activations[row_idx*number_of_samples+col_idx];
		}
	}
}
void delta_compute(float* delta, float* activations, int *labels, int W_rows, int number_of_samples)
{
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((number_of_samples-1)/block.y+1, (W_rows-1)/block.x+1);

    kernel_delta_compute<<<grid, block>>>(delta, activations, labels,  W_rows,  number_of_samples);
	CHECK(cudaDeviceSynchronize());
}

__global__ void kernel_matrix_add(float *input, float *output, int num_rows, int num_cols, float alpha)
{
	int x = blockIdx.x * blockDim.x ;
	int y = blockIdx.y * blockDim.y ;

	int col_idx = x + threadIdx.x;
	int row_idx = y + threadIdx.y;
	

	if(row_idx<num_rows)
	{
		if(col_idx<num_cols)
		{
			output[row_idx*num_cols+col_idx] = output[row_idx*num_cols+col_idx]+alpha*input[row_idx*num_cols+col_idx];
		}
	}
}
void matrix_add(float *input, float *output, int num_rows, int num_cols, float alpha)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((num_cols-1)/block.y+1, (num_rows-1)/block.x+1);
    kernel_matrix_add<<<grid, block>>>(input, output, num_rows,  num_cols,  alpha);
	CHECK(cudaDeviceSynchronize());
}


void matrix_transpose(float *input, float *output, int num_rows, int num_cols);

void matrix_multiply(float *M, float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols);

void softmax(float* activations, int rows, int cols);

float accuracy(float* activations, int* labels, int rows, int cols);

float cross_entropy_loss(float* activations, int* labels, int rows, int cols);

void delta_compute(float* delta, float* activations, int *labels, int W_rows, int number_of_samples);

void matrix_add(float *input, float *output, int num_rows, int num_cols, float alpha);

#endif


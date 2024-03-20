#ifndef _UTILS_H_
#define _UTILS_H_


#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))

#define exp 2.718281828
void weight_initialization(float* W, int W_rows, int W_columns)
{
    int n = W_rows*W_columns;
    for(int i=0; i<n; i++)
    {
        W[i] = RAND_FLOAT();
    }
}

void matrix_transpose(float *input, float *output, int num_rows, int num_cols)
{
	for(int row_idx=0; row_idx<num_rows; row_idx++)
	{
		for(int col_idx=0; col_idx<num_cols; col_idx++)
		{
			output[col_idx*num_rows+row_idx] = input[row_idx*num_cols+col_idx];
		}
	}
}
void matrix_multiply( float *M,  float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols)
{
    for(int row=0; row<M_rows; row++)
    {
        for(int col=0; col<N_cols; col++)
        {
            float Pvalue = 0;
            for(int k=0; k<M_cols; k++)
            {
                Pvalue += M[row*M_cols+k] * N[k*N_cols+col];
            }
            P[row*N_cols+col] = Pvalue;
        }
    }
}
void softmax(float* activations, int rows, int cols)
{
	for(int col_idx=0; col_idx<cols; col_idx++)
	{
		float max=-100000000;
		float sum=0;
		for(int row=0;row<rows;row++)
		{
			if(activations[row*cols+col_idx]>max)
				max=activations[row*cols+col_idx];
		}
		for(int row=0;row<rows;row++)
		{
			activations[row*cols+col_idx]=pow(exp,activations[row*cols+col_idx]-max);
			sum+=activations[row*cols+col_idx];
		}
//		printf("sum:%f\n",sum);
		for(int row=0;row<rows;row++)
		{
//			printf("sotf1: %f\n",(activations[row*cols+col_idx]));
			activations[row*cols+col_idx]=activations[row*cols+col_idx]/sum;
//			printf("sotf2: %f\n",activations[row*cols+col_idx]);

		}
	}
}
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
					loss -=1 * log(activations[row_idx*cols+col_idx]+0.1);
				else
					loss -=1 * log(activations[row_idx*cols+col_idx]);
			}
		}
	}
	return loss/(cols);
}

void delta_compute(float* delta, float* activations, int *labels, int W_rows, int number_of_samples)
{
	for(int col_idx=0; col_idx<number_of_samples; col_idx++)
	{
		for(int row_idx=0; row_idx<W_rows; row_idx++)
		{
			if(row_idx==labels[col_idx])
			{
				delta[row_idx*number_of_samples+col_idx] = activations[row_idx*number_of_samples+col_idx]-1;
			}
			else
				delta[row_idx*number_of_samples+col_idx] = activations[row_idx*number_of_samples+col_idx];
//			printf("delta:%f\n",delta[row_idx*number_of_samples+col_idx]);
		}
	}
}
void matrix_add(float *input, float *output, int num_rows, int num_cols, float alpha)
{
	for(int row_idx=0; row_idx<num_rows; row_idx++)
	{
		for(int col_idx=0; col_idx<num_cols; col_idx++)
		{
//			printf("w1:%f \n",input[row_idx*num_cols+col_idx]);
			output[row_idx*num_cols+col_idx] = output[row_idx*num_cols+col_idx]+alpha*input[row_idx*num_cols+col_idx];

		}
	}
//	printf("alpha:%f\n",alpha);
}
void matrix_transpose(float *input, float *output, int num_rows, int num_cols);

void matrix_multiply(float *M, float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols);

void softmax(float* activations, int rows, int cols);

float accuracy(float* activations, int* labels, int rows, int cols);

float cross_entropy_loss(float* activations, int* labels, int rows, int cols);

void delta_compute(float* delta, float* activations, int *labels, int W_rows, int number_of_samples);

void matrix_add(float *input, float *output, int num_rows, int num_cols, float alpha);

#endif


#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void compute_outputDW_kernel( double *weight_output, const double *output_fm_hidden,  const int nHidden,const int T_size,const double *direction, const int nT, const int *Time, const int * index,  const double decay1, const double decay2,const double V0_param)
{

  //determine the id of the thread
	int hidden_id = blockIdx.x*blockDim.x +threadIdx.x;
	if(hidden_id > nHidden - 1 ) {return;}
	int t_hiddenid = hidden_id*T_size;
	int curT = 0;
	int time_value = Time[curT];
	int curt = 1;

	double Mn = 0;
	double Sn = 0;
	double tmp = 0;


	while(curT < nT)
	{
		tmp = V0_param * output_fm_hidden[t_hiddenid + curt - 1];
		Mn = decay1*Mn + tmp;
		Sn = decay2*Sn + tmp;

		while(curt == time_value)
		{
			weight_output[hidden_id+(index[curT]-1)*nHidden] = (Mn - Sn)*direction[curT];
			curT = curT + 1;
			if(curT >= nT)
			{
				break;
			}
			time_value = Time[curT];
		}
		curt = curt + 1;
	}

}

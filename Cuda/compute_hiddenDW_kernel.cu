#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void compute_hiddenDW_kernel( double *weight_hidden, const double *u, const double * input, const int nInput, const int nHidden,const int T_size, const int nT, const int *Time, const double * weight_toutput,const double* direction,  const double *decay1, const double *decay2, const double threshold,const double V0_param,const double a1)
{

  //determine the id of the thread
	int input_id = blockIdx.x*blockDim.x + threadIdx.x;
	int hidden_id = blockIdx.y;
	if(input_id > nInput - 1 ) {return;}
	if(hidden_id > nHidden - 1 ) {return;}
	int t_inputid = input_id*T_size;
	int t_hiddenid = hidden_id*T_size;
	int curT = 0;
	int curt = 1;
	int time_value = Time[curT];

	double Mn = 0;
	double Sn = 0;
	double Upre = 0;
	double Mpre = 0;
	double Spre = 0;
	double Epre = 0;
	double h = 0;
	double tmp = 0;
	double DeltaW = 0;

	// double Mtmp = 0;

	while(curT < nT)
	{
		Epre = decay1[0]*Epre + h*threshold*Upre;
		tmp = input[t_inputid + curt - 1];
		Mpre = decay1[0]*Mpre + tmp;
		Spre = decay2[0]*Spre + tmp;
		Upre = Mpre - Spre -Epre;
		//h function
		h = u[t_hiddenid + curt - 1];
		if(fabs(h-threshold)<(0.5*a1))
		{
			h = 1/a1;
		}
		else
		{
			h = 0;
		}

		tmp = V0_param*h*Upre;
		Mn = decay1[1]*Mn + tmp;
		Sn = decay2[1]*Sn + tmp;
		// if(h != 0)
		// {
		// 	Mtmp = h;
		// }
		while(curt == time_value)
		{
			DeltaW = DeltaW + direction[curT]*weight_toutput[curT*nHidden + hidden_id]*(Mn - Sn);
			curT = curT + 1;
			if(curT >= nT)
			{
				break;
			}
			time_value = Time[curT];
		}
		curt = curt + 1;
	}
	weight_hidden[input_id + hidden_id*nInput] = DeltaW;
	// weight_hidden[input_id + hidden_id*nInput] = Mtmp;
}

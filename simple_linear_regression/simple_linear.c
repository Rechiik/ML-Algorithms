#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
	float m;
	float c;
} regressionEq;

float hypoFunc(regressionEq eq, float x){
	float c = eq.c;
	float m = eq.m;
	float y = m*x + c;
	return y;
}

void gradientDescent(regressionEq *eq, float x[], float Y[], int m, float lr){
	float dJ_dtheta0 = 0;
	float dJ_dtheta1 = 0;
	for (int i=0;i<m;i++){
		dJ_dtheta0 += hypoFunc(*eq,x[i]) - Y[i];
		dJ_dtheta1 += (hypoFunc(*eq,x[i]) - Y[i])*x[i];
	}
	eq->m = eq->m - lr*dJ_dtheta1;
	eq->c = eq->c - lr*dJ_dtheta0;

}

float costFunction(regressionEq eq, float x[], float Y[], int m){
	float cost = 0;
	for (int i=0;i<m;i++){
		float yPred = hypoFunc(eq,x[i]);
		float yActual = Y[i];
		cost += pow((yPred - yActual),2);	
	}
	cost = cost/ (2 * m);
	return cost;
}


float *predict(regressionEq eq, float x[], int m){
	float* yHat = malloc(m * sizeof(float));
	for (int i=0;i<m;i++){
		yHat[i] = eq.m*x[i] + eq.c;
	}
	return yHat;
}
int main(){
	regressionEq eq = {.c = 1.5,.m=1};
	float x[] = {0,1,2,3,4,5};
	float Y[] = {4,6,8,10,12,14};
	int m = 6;
	float cost = costFunction(eq, x, Y, m);
	int epochs = 100;
	int counter = 0;
	printf("Cost at epoch %d: %f\n",counter,cost);
	while (counter<epochs){
		gradientDescent(&eq, x, Y, m, 0.01);
		counter++;
		cost = costFunction(eq, x, Y,m);
		if (counter%10==0){
		printf("Cost at epoch %d: %f\n",counter,cost);
		printf("Value of theta_0: %f\nValue of theta_1: %f\n",eq.c,eq.m);
		}
	}
	float testX[] = {5.0,20.0,11.0};
	int n =3;
	float *yPred = predict(eq, testX, n);
	for (int i=0;i<n;i++){
		printf("PREDICTED VALUES FOR X = %f IS y=%f\n",testX[i],*(yPred+i));
	}	
	return 0;
}

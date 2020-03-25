#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<stdbool.h>

#define Pi2  6.28
#define e  2.178
#define FLOAT_APPROACH 100000000000000000000000.1
#define bounds 30000

void generateGaussianCurve(float3 *Points,int hostOfELements);
void generateCubeNetwork(float3 *cubeNetwork,int numberOfCubes,float dim);
int serialKNN(float3 host_queryPoints,float3 *host_constantPoints,int hostOfElements);


int main(int argc,char*argv[]){


	struct timeval startwtime, endwtime;
	int ind = atoi(argv[1]) ;

	int hostOfElements = pow(2,ind);
		// Cubes dimension:
	int host_dim = atoi(argv[2]) ;
	float cubeLeng = atof(argv[3]);

	float3 *host_queryPoints =  malloc(sizeof(float3)*hostOfElements);
	float3 *host_constantPoints = malloc(sizeof(float3)*hostOfElements);
	int *host_nN = (int*)malloc(sizeof(int)*hostOfElements);

	generateGaussianCurve(host_queryPoints,hostOfElements);
	generateGaussianCurve(host_constantPoints,hostOfElements);

	int *serialN = (int*)malloc(sizeof(int)*hostOfElements);
	gettimeofday (&startwtime, NULL);

	for(int j = 0;j<hostOfElements;j++){
		serialN[j] = serialKNN(host_queryPoints[j],host_constantPoints,hostOfElements);
	}

	gettimeofday (&endwtime, NULL);
	double serialTime = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
					                      + endwtime.tv_sec - startwtime.tv_sec);
	printf("serialTime:  %f \n",serialTime);

	return 0;


}











void generateGaussianCurve(float3 *Points,int hostOfElements){
	srand((unsigned int)time(NULL));
	float medianX = 0,medianY=0;
	float stdDeviationX = 0;
	float stdDeviationY = 0;

	for(int i = 0;i<hostOfElements;i++){
		Points[i].x = ((float)rand()/(float)(RAND_MAX));
		Points[i].x = floor(10000*Points[i].x)/10000;

		Points[i].y =  ((float)rand()/(float)(RAND_MAX));
		Points[i].y = floor(10000*Points[i].y)/10000;

		medianX = medianX + Points[i].y;
		medianY = medianY+Points[i].y;
	}

	for(int i = 0;i<hostOfElements;i++){
		stdDeviationX = stdDeviationX + powf((Points[i].x-medianX),2.0);
		stdDeviationY = stdDeviationY + powf((Points[i].y-medianY),2.0);
	}

	stdDeviationX = stdDeviationX/(hostOfElements-1);

	stdDeviationY = stdDeviationY/(hostOfElements-1);

	medianX=medianX/hostOfElements;
	medianY = medianY/hostOfElements;
	float x,y;
	float toPow;
	//Calculate z-axis function-leg :
	for(int i =0;i<hostOfElements;i++){
		x = Points[i].x;
		y = Points[i].y;
		toPow = -(powf(x-medianX,2.0)/2*powf(stdDeviationX,2.0));
		toPow = toPow -(powf(y-medianY,2.0)/2*powf(stdDeviationY,2.0));
		toPow = toPow/FLOAT_APPROACH;

		Points[i].z = powf(e,toPow);
		Points[i].z = (Points[i].z/(Pi2*stdDeviationX*stdDeviationY))*10000000000000000000000000.0 ;
		Points[i].z = floor(10000*Points[i].z)/10000;

		while(Points[i].z >1){Points[i].z = Points[i].z - 0.1;}
	}

}

/** Most efficient way to express cubes: VIA THEIR CENTER-MASS **/
void generateCubeNetwork(float3 *cubeNetwork,int numberOfCubes,float dim){

	int i = 0;
	float x=dim,y=dim,z=dim ;
	while(z<1){
		while(y<1){
			while(x<1){
				cubeNetwork[i].x = x;
				cubeNetwork[i].y = y;
				cubeNetwork[i].z = z;
				x = x + 2*dim ;
				i++ ;

			}
			y = y+2*dim ;
			x = dim;
		}
		z = z+2*dim;
		y = dim;
	}
}








int serialKNN(float3 host_queryPoints,float3 *host_constantPoints,int hostOfElements){

	int valueReturn  ;
	float minDist,currDistance ;


		minDist  =  powf(host_queryPoints.x-host_constantPoints[0].x,2.0);
		minDist += powf(host_queryPoints.y-host_constantPoints[0].y,2.0);
		minDist += powf(host_queryPoints.z-host_constantPoints[0].z,2.0);

		for(int j =0;j<hostOfElements;j++){
			currDistance  =  powf(host_queryPoints.x-host_constantPoints[j].x,2.0);
			currDistance += powf(host_queryPoints.y-host_constantPoints[j].y,2.0);
			currDistance += powf(host_queryPoints.z-host_constantPoints[j].z,2.0);

			if(minDist > currDistance){
				minDist = currDistance ;
				valueReturn = j;
			}

		}




	return valueReturn ;
}

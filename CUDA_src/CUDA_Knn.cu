/**
	Author: Chares Moustakas
	AEM   : 8860
	Prof  : Nikolaos Pitsianis,Dimitrios Floros
	email : <charesmoustakas@gmail.com> <cmoustakas@ece.auth.gr>
	Descr : The below code implements kNN algorithm  in parallel computing concept via CUDA.
**/


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


float cubeLen = 0.25;


__device__ int findNearestFromPrCandidates(float3 querryPoint,int *primaryCandidates,float3 *constantPoints,float3 cube,float cubeLen,int* numOfElements,bool tangentSearch){

	float minDist = powf(querryPoint.x-constantPoints[primaryCandidates[0]].x,2.0);
	minDist += powf(querryPoint.y-constantPoints[primaryCandidates[0]].y,2.0);
	minDist += powf(querryPoint.z-constantPoints[primaryCandidates[0]].z,2.0);
	// DON'T SQRT ... YOU ARE GONNA HAVE A TIME PENALTY

	float currDistance ;
	int counter = 0,returnId;

	while(counter < bounds){
		currDistance =  powf(querryPoint.x-constantPoints[primaryCandidates[counter]].x,2.0);
		currDistance += powf(querryPoint.y-constantPoints[primaryCandidates[counter]].y,2.0);
		currDistance += powf(querryPoint.z-constantPoints[primaryCandidates[counter]].z,2.0);

		if(currDistance < minDist){
			minDist = currDistance ;
			returnId = primaryCandidates[counter];
		}
		counter ++ ;
	}

	/** THERE ARE 6 SURFACES :
	 * 1=> Z = Zcube(+-)cubeLen
	 * 2=> Y = Ycube (+-)cubeLen
	 * 3=> X = Xcube (+-)cubeLen
	 */
	if(!tangentSearch){
		float minSurfDist = fabs((querryPoint.x - cube.x - cubeLen));
		if(minSurfDist>fabs(-querryPoint.x + cube.x - cubeLen)){minSurfDist = fabs(-querryPoint.x + cube.x - cubeLen);}
		if(minSurfDist>fabs(querryPoint.y - cube.y - cubeLen)){ minSurfDist = fabs(querryPoint.y - cube.y - cubeLen);}
		if(minSurfDist>fabs(-querryPoint.y + cube.y - cubeLen)){ minSurfDist = fabs(-querryPoint.y + cube.y - cubeLen);}
		if(minSurfDist>fabs(querryPoint.z - cube.z - cubeLen)){ minSurfDist = fabs(querryPoint.z - cube.z - cubeLen);}
		if(minSurfDist>fabs(-querryPoint.z + cube.z - cubeLen)){ minSurfDist = fabs(-querryPoint.z + cube.z - cubeLen);}

	// Now we got the minimum distance between our point and cube's surfaces
		if(minDist > minSurfDist){returnId = returnId - (int)numOfElements ;}
	}
	//printf("returnId is : %d \n",returnId);
	return returnId ;
}







__device__ int *findPrimaryCandidates(float3 *constantPoints,int* numOfElements,float3 cube,float cubeLen){
	float xCube = cube.x;
	float yCube = cube.y;
	float zCube = cube.z;

	float xConst,yConst,zConst;
	int prim_counter = 0 ;
	int *primaryCandidates = (int*)malloc(bounds*sizeof(int));

	bool inside_cube ;
	for(int i =0;i<(int)numOfElements;i++){
		xConst = constantPoints[i].x ;
		yConst = constantPoints[i].y ;
		zConst = constantPoints[i].z ;

		inside_cube = ((xConst<cubeLen+xCube)&&(xConst>xCube-cubeLen)&&(yConst<yCube+cubeLen)&&(yConst>yCube-cubeLen)&&(zConst<zCube+cubeLen)&&(zConst>zCube-cubeLen));
		if(inside_cube){
			primaryCandidates[prim_counter] = i ;
			prim_counter++;
			
			if(prim_counter>bounds){break;}
		}
	}
	return primaryCandidates;
}






__device__ int searchInTangentCube(int nearestN,float3 querryPoint,float3 *constantPoints,float3 cube,int* numOfElements,float cubeLen){

	printf("IM INSIDE TANGENT CUBE !\n");
	bool tangentSearch = true ;
	int* primaryCandidatesOfTangentCube = findPrimaryCandidates(constantPoints,numOfElements,cube,cubeLen);
	int nearestTangentId = findNearestFromPrCandidates(querryPoint,primaryCandidatesOfTangentCube,constantPoints,cube,cubeLen,numOfElements,tangentSearch);

	float currDistance =  powf(querryPoint.x-constantPoints[nearestN].x,2.0);
	currDistance += powf(querryPoint.y-constantPoints[nearestN].y,2.0);
	currDistance += powf(querryPoint.z-constantPoints[nearestN].z,2.0);

	float newDistance =  powf(querryPoint.x-constantPoints[nearestTangentId].x,2.0);
	newDistance += powf(querryPoint.y-constantPoints[nearestTangentId].y,2.0);
	newDistance += powf(querryPoint.z-constantPoints[nearestTangentId].z,2.0);

	if(newDistance < currDistance){nearestN = nearestTangentId;}
	return nearestN ;

}





__global__  void procKernel(float3 *querryPoints,float3 *constantPoints,int* numOfElements,int *numOfcubes,float3 *cubeNetwork,int*nearestNeig,int*Dim,float*cubeLeng){
	
	
	int unique_id = threadIdx.x + blockIdx.x*blockDim.x;


	//cubeLen = (float)cubeLeng;
	
	int dim = (int)Dim;
	//cubeLen = cubeLen/2;
	bool inside_cube ;
	// Lets check our cube !!
	float xQ = querryPoints[unique_id].x;
	float yQ = querryPoints[unique_id].y;
	float zQ = querryPoints[unique_id].z;

	float xCube,yCube,zCube;
	int my_cubeId;

	for(int i = 0;i < (int)numOfcubes;i++){
		xCube = cubeNetwork[i].x;
		yCube = cubeNetwork[i].y;
		zCube = cubeNetwork[i].z;
		inside_cube = ((xQ<cubeLen+xCube)&&(xQ>xCube-cubeLen)&&(yQ<yCube+cubeLen)&&(yQ>yCube-cubeLen)&&(zQ<zCube+cubeLen)&&(zQ>zCube-cubeLen));
		
		if(inside_cube){
			my_cubeId = i;
			break ;
		}

	}

	int *primaryCandidates  = findPrimaryCandidates(constantPoints,numOfElements,cubeNetwork[my_cubeId],cubeLen);


	int tangentCubesDirections[6] = {1,-1,-dim,dim,dim*dim,-dim*dim};
	bool searchTangent = false ;


	int nearestN = findNearestFromPrCandidates(querryPoints[unique_id],primaryCandidates,constantPoints,cubeNetwork[my_cubeId],cubeLen,numOfElements,searchTangent);
	//printf(" did i found my nearest from pr candidates ? %d \n",unique_id);

	if(nearestN<0){

		nearestN = nearestN + (int)numOfElements;



		for(int i = 0;i<6;i++){


			if((tangentCubesDirections[i] == 1) && (cubeNetwork[my_cubeId].x < 1 - cubeLen)){
				nearestN = searchInTangentCube(nearestN,querryPoints[unique_id],constantPoints,cubeNetwork[my_cubeId + tangentCubesDirections[i]],numOfElements,cubeLen);
			}
			if((tangentCubesDirections[i] == -1)&&(cubeNetwork[my_cubeId].x > cubeLen)){
				nearestN = searchInTangentCube(nearestN,querryPoints[unique_id],constantPoints,cubeNetwork[my_cubeId + tangentCubesDirections[i]],numOfElements,cubeLen);
			}
			if((tangentCubesDirections[i] == -dim)&&(cubeNetwork[my_cubeId].y > cubeLen)){
				nearestN = searchInTangentCube(nearestN,querryPoints[unique_id],constantPoints,cubeNetwork[my_cubeId + tangentCubesDirections[i]],numOfElements,cubeLen);
			}
			if((tangentCubesDirections[i] == dim) && (cubeNetwork[my_cubeId].y < 1 - cubeLen)){
				nearestN = searchInTangentCube(nearestN,querryPoints[unique_id],constantPoints,cubeNetwork[my_cubeId + tangentCubesDirections[i]],numOfElements,cubeLen);
			}
			if((tangentCubesDirections[i] == - dim*dim)&&(cubeNetwork[my_cubeId].z > cubeLen)){
				nearestN = searchInTangentCube(nearestN,querryPoints[unique_id],constantPoints,cubeNetwork[my_cubeId + tangentCubesDirections[i]],numOfElements,cubeLen);
			}
			if((tangentCubesDirections[i] == dim*dim) && (cubeNetwork[my_cubeId].z < 1-cubeLen)){
				nearestN = searchInTangentCube(nearestN,querryPoints[unique_id],constantPoints,cubeNetwork[my_cubeId + tangentCubesDirections[i]],numOfElements,cubeLen);
			}
		}

	}

	nearestNeig[unique_id] = nearestN ;
	if(nearestNeig[unique_id] > (int)numOfElements){printf("boundary error \n");}
}



int main(int argc,char *argv[]){
	if(argc == 3){
	    struct timeval startwtime, endwtime;
	  	double cuda_time;

		int ind = atoi(argv[1]) ;

		int hostOfElements = pow(2,ind);
		// Cubes dimension:
		int host_dim = atoi(argv[2]) ;
		float cubeLeng = cubeLen;

		float* device_cubeLen;
		cudaMalloc(&device_cubeLen,sizeof(float));
		cudaMemcpy(device_cubeLen,&cubeLeng,sizeof(float),cudaMemcpyHostToDevice);
		
		int* device_dim ;
		cudaMalloc(&device_dim,sizeof(int));
		cudaMemcpy(device_dim,&host_dim,sizeof(int),cudaMemcpyHostToDevice);

		// Number of Elements on device
		int* device_numOfElements;
		cudaMalloc(&device_numOfElements,sizeof(int));
		cudaMemcpy(device_numOfElements,&hostOfElements,sizeof(int),cudaMemcpyHostToDevice);

		/** declare  Arrays in host **/

		float3 *host_queryPoints = new float3[hostOfElements];
		float3 *host_constantPoints = new float3[hostOfElements];
		int *host_nN = (int*)malloc(sizeof(int)*hostOfElements);


		generateGaussianCurve(host_queryPoints,hostOfElements);
		generateGaussianCurve(host_constantPoints,hostOfElements);


		float3 *device_queryPoints ;
		float3 *device_constPoints;
		int *device_nN ;

		cudaMalloc(&device_queryPoints,sizeof(float3)*hostOfElements);
		cudaMalloc(&device_constPoints,sizeof(float3)*hostOfElements);
		cudaMalloc(&device_nN,sizeof(int)*hostOfElements);

		cudaMemcpy(device_queryPoints,host_queryPoints,sizeof(float3)*hostOfElements,cudaMemcpyHostToDevice);
		cudaMemcpy(device_constPoints,host_constantPoints,sizeof(float3)*hostOfElements,cudaMemcpyHostToDevice);

		int numOfcubes = host_dim*host_dim*host_dim ;

		float3 *host_cubeNetwork = new float3[numOfcubes];

		int *device_numOfcubes;
		cudaMalloc(&device_numOfcubes,sizeof(int));


		float3 *device_cubeNetwork;
		cudaMalloc(&device_cubeNetwork,sizeof(float3)*numOfcubes);


		
		generateCubeNetwork(host_cubeNetwork,numOfcubes,cubeLeng);

		cudaMemcpy(device_cubeNetwork,host_cubeNetwork,sizeof(float3)*numOfcubes,cudaMemcpyHostToDevice);
		cudaMemcpy(device_numOfcubes,&numOfcubes,sizeof(int),cudaMemcpyHostToDevice);

		gettimeofday (&startwtime, NULL);

		procKernel<<<1,hostOfElements>>>(device_queryPoints,device_constPoints,device_numOfElements,device_numOfcubes,device_cubeNetwork,device_nN,device_dim,device_cubeLen);

		cudaMemcpy(host_nN,device_nN,sizeof(int)*hostOfElements,cudaMemcpyDeviceToHost);
		gettimeofday (&endwtime, NULL);

		  /* get time in seconds */
		cuda_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
				                      + endwtime.tv_sec - startwtime.tv_sec);

		//printf("\n parallel-cuda algorithm completed in : %f seconds. \n",cuda_time);
		printf("%f \n",cuda_time);
		cudaFree(device_nN);
		cudaFree(device_cubeNetwork);
		cudaFree(device_numOfcubes);
		cudaFree(device_queryPoints);
		cudaFree(device_constPoints);
		cudaFree(device_dim);
		cudaFree(device_numOfElements);


		time_t t;
		srand((unsigned) time(&t));


	/**	int serial_nN,randomId,success = 0,fails = 0;

		for(int i=0; i<100 ;i++){
			randomId = rand() % ((int)hostOfElements);
			serial_nN = serialKNN(host_queryPoints[randomId],host_constantPoints,hostOfElements);
				printf("\n random element : %d \n",randomId);
				if(serial_nN != host_nN[randomId]){printf("failed for << %d ---- %d >> \n",serial_nN,host_nN[randomId]);fails++;}
				else{printf("SUCCESSSS !!!!!! for << %d ---- %d >> \n",serial_nN,host_nN[randomId]);success++;}
				printf("querry (%f,%f,%f)  \n",host_queryPoints[randomId].x,host_queryPoints[randomId].y,host_queryPoints[randomId].z);
				printf("cuda : (%f,%f,%f) \n",host_constantPoints[host_nN[randomId]].x,host_constantPoints[host_nN[randomId]].y,host_constantPoints[host_nN[randomId]].z);
				printf("serial : (%f,%f,%f) \n\n",host_constantPoints[serial_nN].x,host_constantPoints[serial_nN].y,host_constantPoints[serial_nN].z);

				printf("\n\n cuda nn : %d \n\n",host_nN[randomId]);
		}


		printf("\n\n TOTAL SUCC: %d | TOTAL FAILS : %d \n\n",success,fails);
**/
	}
	else{ printf("ERROR IN ARGUMENTS ****");}
	
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

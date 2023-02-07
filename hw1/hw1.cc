#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

// function
void mergeArray(float *arr1, float *arr2, float *arr_out, int *l1, int *l2);
void mergeArray2(float *arr1, float *arr2, float *arr_out, int *l1, int *l2);
void separateArray(float *arr1, float *arr2, float *arr_out, int *l1, int *l2);



int main(int argc,char** argv){
	//check input
	if(argc!=4){
		fprintf(stderr,"Must provide 3 parameters\n");
		return -1;
	}

    MPI_Init(&argc,&argv);
	unsigned long long n = atoll(argv[1]); // total data
	char *in_file 	= argv[2];
	char *out_file 	= argv[3];

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // current process
	MPI_Comm_size(MPI_COMM_WORLD, &size); // total process
    // double t1 = MPI_Wtime();
    // if process > number of task shut down unnacessa
    MPI_Comm comm_world;
    if(size>n){
        if(rank>=n){
            MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, rank, &comm_world);
        }else{
            MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comm_world);
        }
        if(comm_world == MPI_COMM_NULL){
            printf("Clear Rank %d\n",rank);
            MPI_Finalize();
            return 0;
        }
        MPI_Comm_rank(comm_world, &rank);
        MPI_Comm_size(comm_world, &size);
    } else{
        comm_world = MPI_COMM_WORLD;
    }
    
    // calculate offset and handle number for each process  
    int basic_num = n / size;
    int remainder = n % size;
    int handle_num;
    if(rank<remainder) {
        handle_num = basic_num + 1;
    } else {
        handle_num = basic_num;
    }
    int offset;
    if(rank<remainder){
        offset = rank * handle_num;
    } else {
        offset = (handle_num+1) * remainder + handle_num * (rank - remainder);
    }

    // pre-calculate information;
    int recv_len = (rank+1<remainder)?handle_num:basic_num;
    int tmp_len = handle_num + recv_len;

    // read the input file
    MPI_File f,fh_out;
    float *data = (float*)malloc(sizeof(float)*n);
    MPI_File_open(comm_world, in_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
	MPI_File_read_at(f,sizeof(float)*offset,data,handle_num,MPI_FLOAT,MPI_STATUS_IGNORE);

    boost::sort::spreadsort::spreadsort(data, data + handle_num); //boost is faster than qsort


    // allocate 2 speace to handle 
    float *data_in = (float*) malloc(sizeof(float)*(basic_num+1));
    float *data_tmp = (float*) malloc(sizeof(float)*(basic_num+1)*2);

    // start sorting
    int last_rank = size - 1;
	bool isEven = (rank%2==0);
    bool allSort = false;
    bool sorted = false;
    bool content_change = false;

    

    while(!allSort){
        // odd phase
        sorted = true;
        content_change = false;
        if(!isEven) {
            MPI_Send(data,handle_num,MPI_FLOAT,rank-1,0,comm_world);
        }
        if(isEven && rank!=last_rank){
            MPI_Recv(data_in,recv_len,MPI_FLOAT,rank+1,0,comm_world,MPI_STATUS_IGNORE);
            if(data[handle_num-1] > data_in[0]){ // sort the array
                sorted = false;
                mergeArray(data,data_in,data_tmp,&handle_num,&recv_len);
                separateArray(data,data_in,data_tmp,&handle_num,&recv_len);
                content_change = true;
                MPI_Send(&content_change, 1, MPI_CXX_BOOL, rank+1, 0,comm_world);
                MPI_Send(data_in, recv_len, MPI_FLOAT, rank+1, 0,comm_world);
            } else {
                content_change = false;
                MPI_Send(&content_change, 1, MPI_CXX_BOOL, rank+1, 0,comm_world);
                sorted = true;
            }
        }
        
        if(!isEven){
            MPI_Recv(&content_change, 1,MPI_CXX_BOOL, rank-1, 0 ,comm_world, MPI_STATUS_IGNORE);
            if(content_change){
                MPI_Recv(data, handle_num,MPI_FLOAT, rank-1, 0 ,comm_world, MPI_STATUS_IGNORE);
            }
        }
        
        // even phase
        content_change = false;
        if(isEven && rank!=0) MPI_Send(data,handle_num,MPI_FLOAT,rank - 1,0,comm_world);
        if(!isEven && rank!=last_rank){
            MPI_Recv(data_in,recv_len,MPI_FLOAT,rank+1,0,comm_world,MPI_STATUS_IGNORE);
            if(data[handle_num-1] > data_in[0]){
                sorted = false;
                mergeArray(data,data_in,data_tmp,&handle_num,&recv_len);
                separateArray(data,data_in,data_tmp,&handle_num,&recv_len);
                content_change = true;
                MPI_Send(&content_change, 1, MPI_CXX_BOOL, rank+1, 0,comm_world);
                MPI_Send(data_in, recv_len, MPI_FLOAT, rank+1, 0,comm_world);
            } else{
                content_change = false;
                MPI_Send(&content_change, 1, MPI_CXX_BOOL, rank+1, 0,comm_world);
                sorted = true;
            }
        }
        if(isEven && rank!=0){
            MPI_Recv(&content_change, 1,MPI_CXX_BOOL, rank-1, 0 ,comm_world, MPI_STATUS_IGNORE);
            if(content_change){
                MPI_Recv(data, handle_num,MPI_FLOAT, rank-1, 0 ,comm_world, MPI_STATUS_IGNORE);
            }
        }
        // check if done
        MPI_Allreduce(&sorted, &allSort, 1, MPI_CXX_BOOL,MPI_LAND,comm_world);
    }


	

    // output data
    MPI_File_open(comm_world, out_file, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_out);
	MPI_File_write_at(fh_out, sizeof(float)*offset, data, handle_num, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh_out);
	free(data);
    free(data_in);
    free(data_tmp);
    MPI_Finalize();
}


// merge 2 array and sort
void mergeArray(float *arr1, float *arr2, float *arr_out, int *l1, int *l2){
    int i = 0;
    int j = 0;
    int k = 0;
    while (i < * l1 && j < * l2) {
        if (arr1[i] < arr2[j]) arr_out[k++] = arr1[i++];
        else arr_out[k++] = arr2[j++];
    }
    while (i < * l1)
        arr_out[k++] = arr1[i++];

    while (j < * l2)
        arr_out[k++] = arr2[j++];
}

void separateArray(float *arr1, float *arr2, float *arr_out, int *l1, int *l2){
    for(int i=0; i< (*l1+*l2) ;i++){
        if(i<*l1) arr1[i] = arr_out[i];
        else arr2[i-*l1] = arr_out[i];
    }
}
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include <omp.h>
#include <emmintrin.h>

int curr_row, curr_col;
omp_lock_t omp_lock;
const int Handle_num = 20;
// pre calculate info
double preInfo1;
double preInfo2;
union Pack {
  alignas(16) double d[2]; // this allows us to read data out by accessing d[2]
  __m128d d2; // put data into vector
};


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {

    if (argc != 9) {
		fprintf(stderr, "must provide exactly 8 arguments!\n");
		return 1;
	}

    MPI_Init(&argc, &argv);
	int rank,size;
	unsigned long long ans;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    int ncpus = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    for(int i = 0; i < width * height; i++){
        image[i] = 0;   // for MPI_Reduce
    }

    curr_row = rank;
    preInfo1 = (upper - lower) / height;
    preInfo2 = (right - left) / width;
    omp_init_lock(&omp_lock);


    #pragma omp parallel num_threads(ncpus)
    {
        int local_handle_num = 0;
        int repeats[2];
        int y_ptr,x_ptr[2],x_start,x_end,x_next;
        double x0_buf[2],y0_buf;
        while(1){
            omp_set_lock(&omp_lock);
            if(curr_row>=height){
                omp_unset_lock(&omp_lock);
                break;
            }
            else{
                // get the curr work from global variable
                y_ptr = curr_row;
                x_start = curr_col;
                // update current state for next work
                if(curr_col+Handle_num>width){
                    local_handle_num = width - x_start;
                    curr_row += size;
                    curr_col  = 0;
                }
                else{
                    local_handle_num = Handle_num;
                    curr_col += Handle_num;
                }
            }
            omp_unset_lock(&omp_lock);
            
            // declare union
            Pack x0,y0,x,y,length_squared,xx,yy;
            // constant
            const double ZERO = 0;
            const double TWO = 2;
            Pack const2;
            const2.d2 = _mm_load1_pd(&TWO);
            // local data
            y0_buf = y_ptr * preInfo1 + lower;
            x_ptr[0] = x_start;
            x_ptr[1] = x_start+1;
            x_next = x_start + 2;
            x_end = x_start + local_handle_num;
            x0_buf[0] = x_ptr[0] * preInfo2 +left;
            x0_buf[1] = x_ptr[1] * preInfo2 +left;
            repeats[0] = 0;
            repeats[1] = 0;
            // upload local data to vector
            y0.d2 = _mm_load_pd1(&y0_buf);
            x0.d2 = _mm_load_pd(x0_buf);
            x.d2  = _mm_setzero_pd();
            y.d2  = _mm_setzero_pd();
            xx.d2 = _mm_setzero_pd();
            yy.d2 = _mm_setzero_pd();
            length_squared.d2 = _mm_setzero_pd();

            // vectorize loop
            while(x_ptr[0]<x_end && x_ptr[1]<x_end){
                while(1){
                    //check current data if pop 
                    length_squared.d2 = _mm_add_pd(xx.d2, yy.d2);
                    if (length_squared.d[0] > 4 || length_squared.d[1] > 4) break;
                    // next iteration
                    y.d2 = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x.d2, y.d2), const2.d2), y0.d2);
                    x.d2 = _mm_add_pd(_mm_sub_pd(xx.d2, yy.d2), x0.d2);
                    yy.d2 = _mm_mul_pd(y.d2, y.d2);
                    xx.d2 = _mm_mul_pd(x.d2, x.d2); 
                    repeats[0]++;
                    repeats[1]++;
                    if (repeats[0] >= iters || repeats[1] >= iters) break;
                }
                if (length_squared.d[0] > 4 || repeats[0] >= iters) {
                    image[y_ptr * width + x_ptr[0]] = repeats[0];
                    repeats[0] = 0;
                    x_ptr[0] = x_next;
                    x_next++;
                    x0_buf[0] = x_ptr[0] * preInfo2 + left;
                    x.d2 = _mm_loadl_pd(x.d2, &ZERO);
                    y.d2 = _mm_loadl_pd(y.d2, &ZERO);  
                    length_squared.d2 = _mm_loadl_pd(length_squared.d2, &ZERO);
                }

                if (length_squared.d[1] > 4 || repeats[1] >= iters) {
                    image[y_ptr * width + x_ptr[1]] = repeats[1];
                    repeats[1] = 0;
                    x_ptr[1] = x_next;
                    x_next++;
                    x0_buf[1] = x_ptr[1] * preInfo2 + left;
                    x.d2 = _mm_loadh_pd(x.d2, &ZERO);
                    y.d2 = _mm_loadh_pd(y.d2, &ZERO);
                    length_squared.d2 = _mm_loadh_pd(length_squared.d2, &ZERO);
                }

                x0.d2 = _mm_load_pd(x0_buf);
                yy.d2 = _mm_mul_pd(y.d2, y.d2);
                xx.d2 = _mm_mul_pd(x.d2, x.d2); 
            }

            // hamdle remain data
            if (x_ptr[0] < x_end) {
                xx.d[0] = x.d[0] * x.d[0];
                yy.d[0] = y.d[0] * y.d[0];
                while (repeats[0] < iters && length_squared.d[0] < 4) {
                    double temp = xx.d[0] - yy.d[0] + x0.d[0];
                    y.d[0] = 2 * x.d[0] * y.d[0] + y0.d[0];
                    x.d[0] = temp;
                    xx.d[0] = x.d[0] * x.d[0];
                    yy.d[0] = y.d[0] * y.d[0];
                    length_squared.d[0] = xx.d[0] + yy.d[0];
                    ++repeats[0];
                }
                image[y_ptr * width + x_ptr[0]] = repeats[0];
            }
            if (x_ptr[1] < x_end) {
                xx.d[1] = x.d[1] * x.d[1];
                yy.d[1] = y.d[1] * y.d[1];
                while (repeats[1] < iters && length_squared.d[1] < 4) {
                    double temp = xx.d[1] - yy.d[1] + x0.d[1];
                    y.d[1] = 2 * x.d[1] * y.d[1] + y0.d[1];
                    x.d[1] = temp;
                    xx.d[1] = x.d[1] * x.d[1];
                    yy.d[1] = y.d[1] * y.d[1];
                    length_squared.d[1] = xx.d[1] + yy.d[1];
                    ++repeats[1];
                }
                image[y_ptr * width + x_ptr[1]] = repeats[1];
            }

        }
    }
    printf("Complete %d\n",rank);
    /* draw and cleanup */
    
    int* ans_image = (int*)malloc(width * height * sizeof(int));
    for(int i = 0; i < width * height; i++){
        ans_image[i] = 0;   // for MPI_Reduce
    }
    MPI_Reduce(image, ans_image, width*height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(rank==0) write_png(filename, iters, width, height, ans_image);

    
    free(image);
    free(ans_image);

    MPI_Finalize();
}


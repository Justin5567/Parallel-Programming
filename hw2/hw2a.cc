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

#include <pthread.h>
#include <emmintrin.h>

//#include <time.h>
//#define BILLION  1000000000L;
// global variable
int num_threads;
int iters,width,height;
double left,right,lower,upper;
int *image;

int curr_row, curr_col;
pthread_mutex_t mutex;
const int Handle_num = 40;


// pre calculate info
double preInfo1;
double preInfo2;


union Pack {
  alignas(16) double d[2]; // this allows us to read data out by accessing d[2]
  __m128d d2; // put data into vector
};


void* func4(void *threadId){
    int local_handle_num = 0;
    int repeats[2];
    int y_ptr,x_ptr[2],x_start,x_end,x_next;
    double x0_buf[2],y0_buf;
    while(1){
        pthread_mutex_lock(&mutex);
        if(curr_row==height){
            pthread_mutex_unlock(&mutex);
            break;
        }
        else{
            // get the curr work from global variable
            y_ptr = curr_row;
            x_start = curr_col;
            // update current state for next work
            if(curr_col+Handle_num>width){
                local_handle_num = width - x_start;
                curr_row += 1;
                curr_col  = 0;
            }
            else{
                local_handle_num = Handle_num;
                curr_col += Handle_num;
            }
        }
        pthread_mutex_unlock(&mutex);
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
                x_next ++;
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

    printf("Function All done\n");
    pthread_exit(NULL);

}


// vertorize
void* func3(void *threadId){
    //int* tid = (int*)threadId;
    //int t = *tid; 
    int local_handle_num = 0;
    int repeats[2];
    int local_row, local_col1, local_col2, local_next_col,local_col_end;
    //printf("Enter function 3\n");
    while(1){
        pthread_mutex_lock(&mutex);
        if(curr_row==height){
            pthread_mutex_unlock(&mutex);
            break;
        }
        else{
            // get the curr work from global variable
            local_row = curr_row;
            local_col1 = curr_col;
            // calculate next position for next job
            if(curr_col+Handle_num>width){
                local_handle_num = width - local_col1;
                curr_row +=1;
                curr_col = 0;
            }
            else{
                local_handle_num = Handle_num;
                curr_col += Handle_num;
            }
        }
        pthread_mutex_unlock(&mutex);
       
        // declare pack union
        Pack x0, y0, x, y, length_squared;
        // set initial value
        local_col2 = local_col1 + 1;
        local_next_col = local_col1 + 2;
        local_col_end = local_col1 + local_handle_num;
        //printf("Current handle Row %d, Col %d ~ %d\n",local_row,local_col1,local_col_end);
        double y0_buf = local_row * (preInfo1) + lower;
        double x0_buf[2];
        x0_buf[0] = local_col1 * (preInfo2) + left;
        x0_buf[1] = local_col2 * (preInfo2) + left;
        repeats[0] = repeats[1] = 0;
        
        const double two = 2;
        const double zero = 0;
        __m128d constant2 = _mm_load_pd(&two);
        __m128d xx,yy,tmp;
        // load data to vector
        y0.d2 = _mm_load_pd1(&y0_buf); //pd1 to both side
        x0.d2 = _mm_load_pd(x0_buf);
        x.d2  = _mm_setzero_pd();
        y.d2  = _mm_setzero_pd();
        length_squared.d2 = _mm_setzero_pd();
        
        // start computing
        
        xx = _mm_setzero_pd();
        yy = _mm_setzero_pd();
        bool work1_done = false;
        bool work2_done = false;
        while(local_col1<local_col_end && local_col2<local_col_end){
            // compute current data
            while(!work1_done && !work2_done){
                if(repeats[0] >= iters || length_squared.d[0] >= 4){
                    work1_done = true;
                    //break;
                }
                if(repeats[1] >= iters || length_squared.d[1] >= 4){
                    work2_done = true;
                    //break;
                }
                if(work1_done || work2_done) break;
                // we can vectorize in this loop
                tmp = _mm_add_pd(_mm_sub_pd(xx, yy), x0.d2);
                y.d2 = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x.d2,y.d2),constant2),y0.d2);
                x.d2 = tmp;
                yy = _mm_mul_pd(y.d2,y.d2);
                xx = _mm_mul_pd(x.d2,x.d2);
                length_squared.d2 = _mm_add_pd(xx,yy);
                ++repeats[0];
                ++repeats[1];

            }
            // if one of the vectorization ends we break out and set new data into the finish work
            // update new block status to allow vectorization continue
            if(work1_done){
                image[local_row*width+local_col1] = repeats[0];
                work1_done = false;
                local_col1 = local_next_col;
                local_next_col++;
                // compute new xbuffer x y lengthand load
                x.d2 = _mm_loadl_pd(x.d2, &zero);
                y.d2 = _mm_loadl_pd(y.d2, &zero);
                x0_buf[0] = local_col1*preInfo2+left;
                length_squared.d2 = _mm_loadl_pd(length_squared.d2, &zero);
                repeats[0] = 0;
                x0.d2 = _mm_load_pd(x0_buf);
            }
            if(work2_done){
                image[local_row*width+local_col2] = repeats[1];
                work2_done = false;
                local_col2 = local_next_col;
                local_next_col++;
                x.d2 = _mm_loadh_pd(x.d2, &zero);
                y.d2 = _mm_loadh_pd(y.d2, &zero);
                x0_buf[1] = local_col1*preInfo2+left;
                length_squared.d2 = _mm_loadh_pd(length_squared.d2, &zero);
                repeats[1] = 0;
                x0.d2 = _mm_load_pd(x0_buf);
            }
        }
        // handle last element
        if (local_col1 < local_col_end) {
            while (repeats[0] < iters && length_squared.d[0] < 4) {
                double temp = x.d[0] * x.d[0] - y.d[0] * y.d[0] + x0.d[0];
                y.d[0] = 2 * x.d[0] * y.d[0] + y0.d[0];
                x.d[0] = temp;
                length_squared.d[0] = x.d[0] * x.d[0] + y.d[0] * y.d[0];
                ++repeats[0];
            }
            image[local_row * width +local_col1] = repeats[0];
        }
        if (local_col2 < local_col_end) {
            while (repeats[1] < iters && length_squared.d[1] < 4) {
                double temp = x.d[1] * x.d[1] - y.d[1] * y.d[1] + x0.d[1];
                y.d[1] = 2 * x.d[1] * y.d[1] + y0.d[1];
                x.d[1] = temp;
                length_squared.d[1] = x.d[1] * x.d[1] + y.d[1] * y.d[1];
                ++repeats[1];
            }
            image[local_row * width +local_col2] = repeats[1];
        }
        
    }
    
    printf("Function All done\n");
    pthread_exit(NULL);
}

// load balance mandelbort set
void* func(void *threadId){
    int* tid = (int*)threadId;
    int t = *tid;
    int thread_row = 0;
    //printf("Enter function 1\n");
    while(1){
        pthread_mutex_lock(&mutex);
        if(curr_row == height){
            pthread_mutex_unlock(&mutex);
            break;
        }
        else{
            thread_row = curr_row;
            ++curr_row;
        }
        pthread_mutex_unlock(&mutex);

        double y0 = thread_row * (preInfo1) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * (preInfo2) + left;
            int repeats = 0;
            double x = 0;
            double y = 0;
            double xx = 0;
            double yy = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = xx - yy + x0;
                y = 2 * x * y + y0;
                x = temp;
                xx = x*x;
                yy = y*y;
                length_squared = xx + yy;
                ++repeats;
            }
            image[thread_row * width + i] = repeats;
            //printf("Current handle Row %d, Col %d ,data %d\n",thread_row,i,repeats);
        }

    }


    pthread_exit(NULL);
}

// calculate mandelbort set in thread
void* func2(void *threadId){
    int* tid = (int*)threadId;
    int t = *tid;    
    for (int j = t; j < height; j+=num_threads) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;
            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
    }
    pthread_exit(NULL);
}


// output png (do not touch)
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
    //struct timespec tt1, tt2;
    //clock_gettime(CLOCK_REALTIME, &tt1);
    if (argc != 9) {
		fprintf(stderr, "must provide exactly 8 arguments!\n");
		return 1;
	}
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    unsigned long long ncpus = CPU_COUNT(&cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    num_threads = ncpus;
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pthread_t threads[num_threads];
    pthread_mutex_init (&mutex, NULL);
    curr_row = 0;

    // pre-calculate info
    preInfo1 = (upper - lower) / height;
    preInfo2 = (right - left) / width;
    int args[num_threads];
    for(int i=0;i<num_threads;i++){
        args[i] = i;        
        pthread_create(&threads[i],NULL,func4,(void*)&args[i]);
    }
    
    for(int i=0;i<num_threads;i++){
        pthread_join(threads[i],NULL);
    }
    
    


    /* draw and cleanup */
    //clock_gettime(CLOCK_REALTIME, &tt2);
    //double time = (tt2.tv_sec - tt1.tv_sec)+ (double)( tt2.tv_nsec - tt1.tv_nsec )/ 1000000000.0;
    //printf("%f \n",time);
    write_png(filename, iters, width, height, image);
    free(image);

    pthread_exit(NULL);
}



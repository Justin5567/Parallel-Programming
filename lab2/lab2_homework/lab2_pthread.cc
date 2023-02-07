#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <pthread.h>

int num_threads;
unsigned long long r,rr,k,ans;


struct argument{
	int ID;
	unsigned long long pixels;
};

void* func(void *threadId){
    argument* arg = static_cast<argument*>(threadId);
	//printf("TEST %d\n",*data);
	unsigned long long pixels = 0;
	for(unsigned long long x = arg->ID;x<r;x+=num_threads){
		unsigned long long y = ceil(sqrtl(rr-x*x));
		pixels +=y;
	}
	// pixels %=k;
	// store data to argument
	arg->pixels = pixels;

    pthread_exit(NULL);
}



int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	rr = r*r;
	
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);


	num_threads = ncpus;
	pthread_t threads[num_threads];
	argument args[num_threads];

	// send data to thread
	for(int i=0;i<num_threads;i++){
		args[i].ID = i;
		args[i].pixels = 0;
		pthread_create(&threads[i],NULL,func,(void*)&args[i]);
	}
	// recv data from thread
	for(int i=0;i<num_threads;i++){
		pthread_join(threads[i],NULL);
		ans+=args[i].pixels;
	}
	ans %= k;
	// for (unsigned long long x = 0; x < r; x++) {
	// 	unsigned long long y = ceil(sqrtl(r*r - x*x));
	// 	pixels += y;
	// 	pixels %= k;
	// }

	printf("%llu\n", (4 * ans) % k);
	pthread_exit(NULL);
}

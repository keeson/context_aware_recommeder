#include "cars_task.h"
#include <iostream>
#include <omp.h>
using namespace std;
using namespace cars;

int main(int argc, char *argv[])
{
    if (argc > 3){
        int num_threads = atoi(argv[2]);
        omp_set_num_threads(num_threads);
    }
    else if (argc >=2){
        omp_set_num_threads(20);
    }
    else {
        cout << "usage:" <<endl;
        cout << "cars input_file  num_threads=20" <<endl;
        return -1;
    }
    srand(time(0));
    Task cars_task;
    cars_task.loadInput(argv[1]);
    cars_task.trainModel();
    cars_task.printModel();
    return 0;
}

#include <iostream>
#include <mpi.h>
#include <pthread.h>
#include <cstdlib>
#include <cmath>

//#define DEBUG

#define TASK_AMOUNT 200
#define ITERATION_COUNT 3
#define SIN_CONST 1000
#define L 15000
#define MIN_TASKS_TO_SEND 3
#define PERCENT 30 // can be 50%
const int STOP_CODE = -1; // Код остановки для потоков

typedef struct arguments{
    int rank;
    int * tasks;
} arguments_t;

int currentTask = 0;
int leftTasks = 0;
int localTasks = 0;
int allDoneTasks = 0;


pthread_mutex_t mutex;

void spaceByRank(int rank) {
    for (int i = 0; i < rank; ++i) {
        std::cout << "\t";
    }
}

void *recvTasks(void *tasksAndRank) {
    arguments_t params = *((arguments_t *) tasksAndRank);
    int rank = params.rank;
    int sendTasks = 0;
    int recvStatus = 0;

    MPI_Status status;
    while (true) {
        MPI_Recv(&recvStatus, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
#ifdef DEBUG
        spaceByRank(rank);
        std::cout << rank << " has recvStatus = " << recvStatus << std::endl;
#endif
        if (recvStatus == STOP_CODE) {
            pthread_exit(nullptr);
        }
        pthread_mutex_lock(&mutex);
        if (leftTasks > MIN_TASKS_TO_SEND) {
            sendTasks = static_cast<int>(leftTasks * PERCENT / 100); //30% can be 50%
            leftTasks -= sendTasks;
            MPI_Send(&sendTasks, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
            MPI_Send(&params.tasks[currentTask + 1], sendTasks, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
            currentTask += sendTasks;
#ifdef DEBUG
            spaceByRank(rank);
            std::cout << sendTasks << "was send from "<< rank << " to " <<  status.MPI_SOURCE << std::endl;
#endif
        } else {
            pthread_mutex_unlock(&mutex);
            sendTasks = 0;
            MPI_Send(&sendTasks, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
#ifdef DEBUG
            spaceByRank(rank);
            std::cout << rank << " NO SEND" << std::endl;
#endif
        }
        pthread_mutex_unlock(&mutex);
    }
}

void execute(const int * tasks, int numTasks, int rank) {

    pthread_mutex_lock(&mutex);
    leftTasks = numTasks;
    int curTask=0;

    while (leftTasks) {
        leftTasks--;
        localTasks++;
        allDoneTasks++;
        double task_size = tasks[curTask];
        pthread_mutex_unlock(&mutex);


        for (int i = 0; i < task_size; ++i) {
            sin(i * SIN_CONST);
        }

        pthread_mutex_lock(&mutex);
#ifdef DEBUG
        spaceByRank(rank);
        std::cout << "Task №" << curTask << " executed by " << rank << std::endl;
        spaceByRank(rank);
        std::cout << "Executed tasks process amount " << allDoneTasks << std::endl;
#endif
        curTask++;
    }

    pthread_mutex_unlock(&mutex);
}


int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    pthread_attr_t attrs;
    pthread_t recv;
    pthread_attr_init(&attrs);
    pthread_mutex_init(&mutex, nullptr);
    pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);
    int processTasksAmount = TASK_AMOUNT / size + ((rank < TASK_AMOUNT % size) ? 1 : 0);
    leftTasks = processTasksAmount;
    std::cout << processTasksAmount << " tasks for №" << rank << std::endl;
    int * tasks = (int *) calloc(processTasksAmount, sizeof(int));
    arguments_t params = {rank, tasks};
    pthread_create(&recv, &attrs, recvTasks, &params);
    pthread_attr_destroy(&attrs);
    bool execMore = true;
    int execMoreTasksAmount = 0;
    double startTime = MPI_Wtime();
    double iterationStartTime = 0;
    double iterationTime = 0;
    for (int num = 0; num < ITERATION_COUNT; ++num) {
        for (int i = 0; i < processTasksAmount; ++i) {
            tasks[i] = abs(100-i%100) * abs(rank - (num % size)) * L;
        }
        iterationStartTime = MPI_Wtime();
        spaceByRank(rank);
        std::cout << "Process №" << rank << " started" << std::endl;
        execute(tasks, processTasksAmount, rank);
        double time =  MPI_Wtime() - iterationStartTime;
        spaceByRank(rank);
        std::cout << rank << " finished" << localTasks << " done " << time << std::endl;
        while (execMore) {
            execMore = false;
            int i = (rank + 1) % size;
            while(i != rank)  {
                spaceByRank(rank);
                std::cout << rank << " send to " << i << std::endl;
                MPI_Send(&rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Recv(&execMoreTasksAmount, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
#ifdef DEBUG
                spaceByRank(rank);
                std::cout << execMoreTasksAmount << " extra tasks for " << rank << std::endl;
#endif
                if (execMoreTasksAmount) { // > 0
                    execMore = true;
                    MPI_Recv(tasks, execMoreTasksAmount, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
                    spaceByRank(rank);
                    std::cout << "Process " << rank << "get " << execMoreTasksAmount << " tasks from " << i << std::endl;
                    execute(tasks, execMoreTasksAmount, rank);
#ifdef DEBUG
                    spaceByRank(rank);
                    std::cout << rank << " finished" << localTasks << " done "<< std::endl;
#endif
                }
                i = (i + 1) % size;
            }

        }
        iterationTime= MPI_Wtime() - iterationStartTime;
        spaceByRank(rank);
        std::cout << rank << " finished iteration with time = " << iterationTime << std::endl;
        double minTime = 0;
        double maxTime = 0;
        MPI_Allreduce(&iterationTime, &minTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&iterationTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==0){
            double disbalance = maxTime - minTime;
            std::cout << "Disbalance = " << disbalance << std::endl;
            std::cout << "Disbalance in % = " << disbalance / maxTime * 100 << std::endl;
        }
        int executed_tasks = 0;
        MPI_Allreduce(&allDoneTasks, &executed_tasks , 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        spaceByRank(rank);
        std::cout << "Executed tasks count = " <<  executed_tasks  << std::endl;
        execMore=true;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Send(&STOP_CODE, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
    if (rank == 0){
        std::cout << "Whole time = " <<  MPI_Wtime() - startTime << std::endl;
    }
    free(tasks);
    pthread_join(recv, nullptr);
    pthread_mutex_destroy(&mutex);
    MPI_Finalize();
    return 0;
}

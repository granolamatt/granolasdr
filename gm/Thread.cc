#include <stdio.h>
#include <pthread.h>
#include "Thread.h"

std::mutex SynchroObjects::synchro_mutex; //Static initialization is needed

namespace gm {
    
void Thread::join() {
    pthread_join(thd, NULL);
}

void Thread::start() {
    //printf("Starting the pthread\n");
    pthread_create (&thd, NULL, &Thread::startThread, this);
}

void* Thread::startThread(void* thd) {
    Runnable* myrunnable = ((Thread*)thd)->myrunnable;
    // if (myrunnable) {
	    myrunnable->setRunning(true);
    // }
    ((Thread*)thd)->run();
    // if (myrunnable) {
        myrunnable->setRunning(false);
    // }
    ((Runnable*)thd)->doCallback((Runnable*)thd);
}

}
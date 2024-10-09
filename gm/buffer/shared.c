#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <semaphore.h>
#include "shared.h"

#define MAX_RETRIES 10
#define SHM_SIZE 4096

union semun {
	int val;
	struct semid_ds *buf;
	ushort *array;
};

/**
 *  initsem() --more-than-inspired by W. Richard Stevens' UNIX Network
 *  Programming 2nd edition, volume 2, lockvsem.c page 295.
 */
int initsem_shared(key_t key, int nsems) /* key from ftok() */
{
	int i;
	union semun arg;
	struct semid_ds buf;
	struct sembuf sb;
	int semid;
	
	semid = semget(key, nsems, IPC_CREAT | IPC_EXCL | 0666);
	
	if (semid >= 0) { /* we got it first */
		sb.sem_op = 1;
		sb.sem_flg = 0;
		arg.val = 1;
		
		for (sb.sem_num = 0; sb.sem_num < nsems; sb.sem_num++) {
			/* do a semop() to "free" the semaphores */
			/* this sets the sem_otime field, as needed below */
			if (semop(semid, &sb, 1) == -1) {
				int e = errno;
				semctl(semid, 0, IPC_RMID); /* clean up */
				errno = e;
				return -1; /* error, check errno */
			}
		}
		
	} else if (errno == EEXIST) { /* someone else got it first */
		int ready = 0;
		
		semid = semget(key, nsems, 0); /* get the id */
		if (semid < 0) return semid; /* error, check errno */
		
		/* wait for other process to initialize the semaphore: */
		arg.buf = &buf;
		for (i = 0; i < MAX_RETRIES && !ready; i++) {
			semctl(semid, nsems-1, IPC_STAT, arg);
			if (arg.buf->sem_otime != 0) {
				ready = 1;
			} else {
				sleep(1);
			}
		}
		if (!ready) {
			errno = ETIME;
			return -1;
		}
	} else {
		return semid; /* error, check errno */
	}
	
	return semid;
}

long get_shared_memory_size(const char *designator) {
    int md;
    struct stat file_stat;
    

    md = shm_open(designator, O_CREAT | O_RDWR, 0);
    if (md == -1) {
       printf("Got an error opening shm\n");
       return -1;
    }
    
    if (fstat(md, &file_stat)) {
    	perror("fstat error\n");
    	return -1;
    }
    
    return file_stat.st_size;
}


char *map_shared_memory(const char *designator, long size) {
    int md;

    md = shm_open(designator, O_CREAT | O_RDWR, 0600);
    if (md == -1) {
       printf("Got an error opening shm\n");
       exit(1);
    }
    int ferror = ftruncate(md, size);
    
    if (ferror == -1) {
    	printf("Error in ftruncate\n");
    }
    
    char* ret = (char *) mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, md, 0);

    return ret;
}

int close_shared_memory(const char *designator) {
    return shm_unlink(designator);
}

int unmap_shared_memory(char* memory_address, int size) {
    return munmap(memory_address, size);
}

/*
  can use:
   if (sem_init(mutex, 1, 1) < 0) {
    perror("semaphore initialization");
    exit(1);
  }
*/

sem_t* open_posix_sem(const char *designator) {
    sem_t* sem_id;
    sem_id = sem_open(designator, O_CREAT, 0600, 1);

    if (sem_id == SEM_FAILED) {
        perror("failed on sem_open");
        return NULL;
    }
    return sem_id;
}

int close_posix_sem(const char *designator) {
/*    sem_t* sem_id;
    sem_id = sem_open(designator, O_CREAT, 0600, 0);

    if (sem_id == SEM_FAILED) {
        perror("failed on sem_open");
        return 1;
    }*/
   return sem_unlink(designator);
    
}

int posix_synchronized(sem_t *sem_id) {

    if (sem_id) {
        if (sem_wait(sem_id) < 0) {
        	perror("sem_wait");
            return -2;
        }
    } else {
        return -1;
    }
    return 0;
}

int posix_unlock(sem_t *sem_id) {
    if (sem_id) {
		if (sem_post(sem_id) < 0) {
            perror("sem_wait");
            return -1;
        } 
    }
    return 0;
}

int init_sysv_keys(SysVStruct* status, const char *lockFile) {
	key_t key;
	key_t keyval;
	
	if ((key = ftok(lockFile, 'J')) == -1) {
		return -1;
	}
		
	/* grab the semaphore set created by seminit.c: */
	if ((status->semaphore_id = initsem_shared(key, 1)) == -1) {
		return -1;
	}

	if ((keyval = ftok(lockFile, 'K')) == -1) {
		return -1;
	}
	
	/* connect to (and possibly create) the shared memory segment: */
	if ((status->memory_id = shmget(keyval, SHM_SIZE, 0666 | IPC_CREAT)) == -1) {
		return -1;
	}
	
	return 0;
}

int close_sysv_sem(const char *designator) {
/*    sem_t* sem_id;
    sem_id = sem_open(designator, O_CREAT, 0600, 0);

    if (sem_id == SEM_FAILED) {
        perror("failed on sem_open");
        return 1;
    }*/
   return sem_unlink(designator);
    
}


int system_wait(SysVStruct* status)
{
	struct sembuf sb;
	
	sb.sem_num = 0;
	sb.sem_op = -1; /* set to allocate resource */
	sb.sem_flg = 0;
	
	return semop(status->semaphore_id, &sb, 1);
	
}

int system_notify(SysVStruct* status)
{

	struct sembuf sb;
	
	sb.sem_num = 0;
	sb.sem_op = 1;
	sb.sem_flg = 0;
	
	return semop(status->semaphore_id, &sb, 1);
}

int system_notifyAll(SysVStruct* status)
{

	struct sembuf sb;

	int numProc = semctl(status->semaphore_id, 0, GETNCNT);
	
	if (numProc) {
		sb.sem_num = 0;
		sb.sem_op = numProc;
		sb.sem_flg = 0;

		if (semop(status->semaphore_id, &sb, 1) == -1) {
			return -1;
		}
	}
	
	return numProc;
}


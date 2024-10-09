#ifndef SHARED_MEM_FUNC_H_
#define SHARED_MEM_FUNC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <semaphore.h>

typedef struct {
  int 		semaphore_id;
  int 		memory_id;
} SysVStruct;


long get_shared_memory_size(const char *designator);
char *map_shared_memory(const char *designator, long size);
int close_shared_memory(const char *designator);
int unmap_shared_memory(char* memory_address, int size);
sem_t* open_posix_sem(const char *designator);
int close_posix_sem(const char *designator);
int posix_synchronized(sem_t *sem_id);
int posix_unlock(sem_t *sem_id);
int init_sysv_keys(SysVStruct* status, const char *designator);
int close_sysv_sem(const char *designator);
int system_wait(SysVStruct* status);
int system_notify(SysVStruct* status);
int system_notifyAll(SysVStruct* status);

#ifdef __cplusplus
}
#endif

#endif /* SHARED_MEM_FUNC_H_ */


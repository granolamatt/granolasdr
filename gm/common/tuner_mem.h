#ifndef TUNERMEM_H_
#define TUNERMEM_H_

#define STRINGSIZE 16384
#define BUFFERSIZE 67108864
#define MAXLOCKSIZE 2048


#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float       x;
    float       y;
} ComplexFloat;

typedef struct
{
    short       x;
    short       y;
} ComplexShort;

typedef struct
{
    ComplexShort    samples[BUFFERSIZE];
} RawSampleHolder;

/**
 *   This is a struct to hold raw a/d and d/a samples in memory
*/
typedef struct
{
    char                    version[STRINGSIZE];
    char                    description[STRINGSIZE];
    char                    lock[MAXLOCKSIZE];
    long                    buffer_size;
    double              frequency;
    long             sample_rate;
    long                    current_sample;
    RawSampleHolder sample_holder;
} RawTunerStruct;


#ifdef __cplusplus
}
#endif

#endif /* TUNERMEM_H_ */

#ifndef _GM_CUDA_HOSTCUDA_
#define _GM_CUDA_HOSTCUDA_

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

#define cuda_check_error( err ) gm::cuda::throw_on_cuda_error(err, __FILE__, __LINE__);

namespace gm {
namespace cuda {

void throw_on_cuda_error(cudaError_t code, const char *file, int line);


}
}


#endif //_GM_CUDA_HOSTCUDA_

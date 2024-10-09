#include <thrust/complex.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

#include "gm/cuda/HostCuda.h"

namespace gm {
namespace cuda {

void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::stringstream ss;
        ss << file << "(" << line << ")";
        std::string file_and_line;
        ss >> file_and_line;
        throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
    }
}

}
}



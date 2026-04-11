
#ifndef RUNTIME_CORE_HPP
#define RUNTIME_CORE_HPP

#include "tensors_struct.h"

#ifdef _WIN32
#define EXPOSE_FUNCTION __declspec(dllexport)
#else
#define EXPOSE_FUNCTION __attribute__((visibility("default")))
#endif

extern "C" EXPOSE_FUNCTION int runtime_initialization_with_args(int length, char **keys, void **values);

extern "C" EXPOSE_FUNCTION int runtime_initialization();

extern "C" EXPOSE_FUNCTION int runtime_model_loading(const char *model_path);

extern "C" EXPOSE_FUNCTION int send_input(tensors_struct *input_tensors);

extern "C" EXPOSE_FUNCTION int receive_output(tensors_struct **output_tensors);

// Return a received output buffer back to the runtime.
// Use this instead of deep_free_tensors_struct() to enable buffer pool reuse.
// Falls back to deep_free_tensors_struct() when the pool is not active.
extern "C" EXPOSE_FUNCTION void runtime_return_output(tensors_struct *output);

extern "C" EXPOSE_FUNCTION int runtime_destruction();

extern "C" EXPOSE_FUNCTION const char *runtime_error_message();

extern "C" EXPOSE_FUNCTION const char *runtime_version();

extern "C" EXPOSE_FUNCTION const char *runtime_name();


#endif // RUNTIME_CORE_HPP
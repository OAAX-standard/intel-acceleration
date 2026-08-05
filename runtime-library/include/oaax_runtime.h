#ifndef OAAX_RUNTIME_INTERFACE_H
#define OAAX_RUNTIME_INTERFACE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define OAAX_EXPORT __declspec(dllexport)
#else
#define OAAX_EXPORT __attribute__((visibility("default")))
#endif

typedef enum RuntimeStatus {
  RUNTIME_STATUS_SUCCESS = 0,
  RUNTIME_STATUS_ERROR = 1,
  RUNTIME_STATUS_NOT_INITIALIZED = 2,
  RUNTIME_STATUS_ALREADY_INITIALIZED = 3,
  RUNTIME_STATUS_MODEL_NOT_LOADED = 4,
  RUNTIME_STATUS_INVALID_ARGUMENT = 5,
  RUNTIME_STATUS_INVALID_MODEL = 6,
  RUNTIME_STATUS_FILE_NOT_FOUND = 7,
  RUNTIME_STATUS_OUT_OF_MEMORY = 8,
  RUNTIME_STATUS_INVALID_TENSOR = 9,
  RUNTIME_STATUS_TENSOR_SHAPE_MISMATCH = 10,
  RUNTIME_STATUS_TENSOR_TYPE_MISMATCH = 11,
  RUNTIME_STATUS_NO_OUTPUT_AVAILABLE = 12,
  RUNTIME_STATUS_INFERENCE_ERROR = 13,
  RUNTIME_STATUS_NOT_IMPLEMENTED = 14,
  RUNTIME_STATUS_TIMEOUT = 15,
  RUNTIME_STATUS_DEVICE_ERROR = 16,
  RUNTIME_STATUS_UNKNOWN_ERROR = 17,
  RUNTIME_STATUS_INVALID_MODEL_ID = 18
} RuntimeStatus;

typedef enum TensorElementType {
  DATA_TYPE_UNDEFINED = 0,
  DATA_TYPE_FLOAT = 1,
  DATA_TYPE_UINT8 = 2,
  DATA_TYPE_INT8 = 3,
  DATA_TYPE_UINT16 = 4,
  DATA_TYPE_INT16 = 5,
  DATA_TYPE_INT32 = 6,
  DATA_TYPE_INT64 = 7,
  DATA_TYPE_STRING = 8,
  DATA_TYPE_BOOL = 9,
  DATA_TYPE_FLOAT16 = 10,
  DATA_TYPE_DOUBLE = 11,
  DATA_TYPE_UINT32 = 12,
  DATA_TYPE_UINT64 = 13,
  DATA_TYPE_COMPLEX64 = 14,
  DATA_TYPE_COMPLEX128 = 15,
  DATA_TYPE_BFLOAT16 = 16,
  DATA_TYPE_FLOAT8E4M3FN = 17,
  DATA_TYPE_FLOAT8E4M3FNUZ = 18,
  DATA_TYPE_FLOAT8E5M2 = 19,
  DATA_TYPE_FLOAT8E5M2FNUZ = 20,
  DATA_TYPE_UINT4 = 21,
  DATA_TYPE_INT4 = 22,
  DATA_TYPE_FLOAT4E2M1 = 23,
  DATA_TYPE_FLOAT8E8M0 = 24,
  DATA_TYPE_UINT2 = 25,
  DATA_TYPE_INT2 = 26
} TensorElementType;

typedef struct TensorDescriptor {
  char *name;
  TensorElementType data_type;
  int rank;
  int *shape;
  size_t data_size;
  void *data;
} TensorDescriptor;

typedef struct Tensors {
  int id;
  int num_tensors;
  TensorDescriptor *tensors;
} Tensors;

typedef struct Config {
  int length;
  const char **keys;
  const char **values;
} Config;

typedef struct ModelConfig {
  const char *file_path;
  const unsigned char *model_data;
  size_t model_size;
  Config config;
} ModelConfig;

OAAX_EXPORT RuntimeStatus runtime_init(Config config);
OAAX_EXPORT RuntimeStatus runtime_load_models(int num_models,
                                              const ModelConfig *model_configs);
OAAX_EXPORT RuntimeStatus runtime_enqueue_input(int model_id,
                                                Tensors *input_tensors);
OAAX_EXPORT RuntimeStatus runtime_retrieve_output(int *model_id,
                                                  Tensors **output_tensors,
                                                  int timeout_ms);
OAAX_EXPORT RuntimeStatus runtime_cleanup(void);
OAAX_EXPORT const char *runtime_get_error(void);
OAAX_EXPORT const char *runtime_get_version(void);
OAAX_EXPORT const char *runtime_get_name(void);
OAAX_EXPORT const char *runtime_get_info(void);

#ifdef __cplusplus
}
#endif

#endif  // OAAX_RUNTIME_INTERFACE_H

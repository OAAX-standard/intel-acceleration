#ifndef TENSORS_STRUCT_H
#define TENSORS_STRUCT_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tensor data types matching OAAX standard
typedef enum tensor_data_type {
  UNDEFINED = 0,
  DATA_TYPE_FLOAT = 1,
  DATA_TYPE_UINT8 = 2,
  DATA_TYPE_INT8 = 3,
  DATA_TYPE_UINT16 = 4,
  DATA_TYPE_INT16 = 5,
  DATA_TYPE_INT32 = 6,
  DATA_TYPE_INT64 = 7,
  DATA_TYPE_STRING = 8,
  DATA_TYPE_BOOL = 9,
  DATA_TYPE_DOUBLE = 11,
  DATA_TYPE_UINT32 = 12,
  DATA_TYPE_UINT64 = 13,
} tensor_data_type;

// Struct to hold multiple tensors — field order must match the OAAX standard
// interface (deps/tools/c-utilities/include/tensors_struct.h).
typedef struct tensors_struct {
  size_t num_tensors;            // Number of tensors
  char **names;                  // Array of tensor names
  tensor_data_type *data_types;  // Array of data types
  size_t *ranks;                 // Array of tensor ranks (number of dimensions)
  size_t **shapes;               // Array of shape arrays
  void **data;                   // Array of data pointers
} tensors_struct;

// Get byte size for a given data type (returns 0 for unsupported/variable
// types)
static inline int get_data_type_byte_size(tensor_data_type type) {
  switch (type) {
    case DATA_TYPE_FLOAT:
      return 4;
    case DATA_TYPE_UINT8:
      return 1;
    case DATA_TYPE_INT8:
      return 1;
    case DATA_TYPE_UINT16:
      return 2;
    case DATA_TYPE_INT16:
      return 2;
    case DATA_TYPE_INT32:
      return 4;
    case DATA_TYPE_INT64:
      return 8;
    case DATA_TYPE_BOOL:
      return 1;
    case DATA_TYPE_DOUBLE:
      return 8;
    case DATA_TYPE_UINT32:
      return 4;
    case DATA_TYPE_UINT64:
      return 8;
    case DATA_TYPE_STRING:
      return 0;  // variable-length
    case UNDEFINED:
      return 0;
    default:
      return 0;
  }
}

// Allocate tensors_struct with given number of tensors
static inline tensors_struct *allocate_tensors_struct(size_t num_tensors) {
  tensors_struct *ts = (tensors_struct *)malloc(sizeof(tensors_struct));
  if (!ts) return NULL;

  ts->num_tensors = num_tensors;
  ts->names = (char **)calloc(num_tensors, sizeof(char *));
  ts->ranks = (size_t *)calloc(num_tensors, sizeof(size_t));
  ts->shapes = (size_t **)calloc(num_tensors, sizeof(size_t *));
  ts->data_types =
      (tensor_data_type *)calloc(num_tensors, sizeof(tensor_data_type));
  ts->data = (void **)calloc(num_tensors, sizeof(void *));

  return ts;
}

// Free tensors_struct and all its contents
static inline void deep_free_tensors_struct(tensors_struct *ts) {
  if (!ts) return;

  for (size_t i = 0; i < ts->num_tensors; i++) {
    if (ts->names && ts->names[i]) free(ts->names[i]);
    if (ts->shapes && ts->shapes[i]) free(ts->shapes[i]);
    if (ts->data && ts->data[i]) free(ts->data[i]);
  }

  if (ts->names) free(ts->names);
  if (ts->ranks) free(ts->ranks);
  if (ts->shapes) free(ts->shapes);
  if (ts->data_types) free(ts->data_types);
  if (ts->data) free(ts->data);

  free(ts);
}

#ifdef __cplusplus
}
#endif

#endif  // TENSORS_STRUCT_H

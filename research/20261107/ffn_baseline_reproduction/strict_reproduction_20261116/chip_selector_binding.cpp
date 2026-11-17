#include <ATen/ATen.h>
#include <torch/library.h>

namespace chipmunk {
void topk_indices(at::Tensor, at::Tensor, at::Tensor, double, int64_t, double);
void copy_indices(at::Tensor, at::Tensor, at::Tensor, at::Tensor);
}

// Only link the two unchanged author CUDA translation units used by the MLP.
TORCH_LIBRARY(chipmunk, m) {
  m.def("topk_indices(Tensor activation, Tensor(a!) indices, Tensor(b!) counts, float sparsity, int multiple_of, float random_amount) -> ()",
        &chipmunk::topk_indices);
  m.def("copy_indices(Tensor input, Tensor(a!) cache, Tensor indices, Tensor counts) -> ()",
        &chipmunk::copy_indices);
}

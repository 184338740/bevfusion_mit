#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA forward launcher (xmy)
void bev_pool_sum_count_xmy(int b, int d, int h, int w, int n, int c, int n_intervals,
                            const float* x, const int* geom_feats,
                            const int* interval_starts, const int* interval_lengths,
                            float* out_sum, float* out_count);

// CUDA backward launcher (xmy)
void bev_pool_sum_count_xmy_grad(int b, int d, int h, int w, int n, int c, int n_intervals,
                                 const float* out_grad, const int* geom_feats,
                                 const int* interval_starts, const int* interval_lengths,
                                 float* x_grad);

// Forward Python interface
std::tuple<at::Tensor, at::Tensor> bev_pool_sum_count_forward(
    const at::Tensor _x,
    const at::Tensor _geom_feats,
    const at::Tensor _interval_lengths,
    const at::Tensor _interval_starts,
    int b, int d, int h, int w) {

  int n = _x.size(0);
  int c = _x.size(1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_x));
  const float* x = _x.data_ptr<float>();
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options = torch::TensorOptions().dtype(_x.dtype()).device(_x.device());
  at::Tensor out_sum = torch::zeros({b, d, h, w, c}, options);
  at::Tensor out_count = torch::zeros({b, d, h, w}, options);
  float* out_sum_ptr = out_sum.data_ptr<float>();
  float* out_count_ptr = out_count.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bev_pool_sum_count_xmy(b, d, h, w, n, c, n_intervals,
                         x, geom_feats, interval_starts, interval_lengths,
                         out_sum_ptr, out_count_ptr);

  return std::make_tuple(out_sum, out_count);
}

// Backward Python interface (grad_out_count is ignored)
at::Tensor bev_pool_sum_count_backward(
    const at::Tensor _out_grad,
    const at::Tensor _geom_feats,
    const at::Tensor _interval_lengths,
    const at::Tensor _interval_starts,
    int b, int d, int h, int w) {

  int n = _geom_feats.size(0);
  int c = _out_grad.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  const float* out_grad = _out_grad.data_ptr<float>();
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options = torch::TensorOptions().dtype(_out_grad.dtype()).device(_out_grad.device());
  at::Tensor x_grad = torch::zeros({n, c}, options);
  float* x_grad_ptr = x_grad.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bev_pool_sum_count_xmy_grad(b, d, h, w, n, c, n_intervals,
                              out_grad, geom_feats, interval_starts, interval_lengths,
                              x_grad_ptr);

  return x_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_pool_sum_count_forward", &bev_pool_sum_count_forward,
        "bev_pool_sum_count_forward (xmy)");
  m.def("bev_pool_sum_count_backward", &bev_pool_sum_count_backward,
        "bev_pool_sum_count_backward (xmy)");
}
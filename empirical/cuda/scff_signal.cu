#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void scff_signal_kernel(
    const float* __restrict__ z, const float* __restrict__ zpos,
    float* __restrict__ s_perp, float* __restrict__ goodness,
    const float tau, const int B, const int C) {
  extern __shared__ float sm[];
  float* zi  = sm;            // [C]
  float* zpi = sm + C;        // [C]
  float* acc = sm + 2*C;      // [C]  (reused to hold s_i)
  float* sc  = sm + 3*C;      // [B]  scores
  float* red = sm + 3*C + B;  // [blockDim.x]
  const int i = blockIdx.x, t = threadIdx.x, nt = blockDim.x;

  for (int c = t; c < C; c += nt) { zi[c]=z[i*C+c]; zpi[c]=zpos[i*C+c]; acc[c]=0.f; }
  __syncthreads();

  float lmax = -1e30f;
  for (int j = 0; j < B; ++j) {
    float partial = 0.f;
    for (int c = t; c < C; c += nt) partial += zi[c]*z[j*C+c];
    red[t]=partial; __syncthreads();
    for (int s=nt/2; s>0; s>>=1) { if (t<s) red[t]+=red[t+s]; __syncthreads(); }
    if (t==0) sc[j]=red[0]/tau;
    __syncthreads();
    lmax = fmaxf(lmax, sc[j]);
  }
  float sumexp = 0.f;
  for (int j = 0; j < B; ++j) sumexp += __expf(sc[j]-lmax);
  const float lse = lmax + logf(sumexp);

  for (int c = t; c < C; c += nt) {
    float a = 0.f;
    for (int j = 0; j < B; ++j) a += __expf(sc[j]-lse) * z[j*C+c];
    acc[c] = a;
  }
  __syncthreads();

  float partial = 0.f;
  for (int c = t; c < C; c += nt) { float si=zpi[c]-acc[c]; acc[c]=si; partial+=zi[c]*si; }
  red[t]=partial; __syncthreads();
  for (int s=nt/2; s>0; s>>=1) { if (t<s) red[t]+=red[t+s]; __syncthreads(); }
  const float dot_zs = red[0];

  for (int c = t; c < C; c += nt) s_perp[i*C+c] = acc[c] - zi[c]*dot_zs;

  float gp = 0.f;
  for (int c = t; c < C; c += nt) gp += zi[c]*zpi[c];
  red[t]=gp; __syncthreads();
  for (int s=nt/2; s>0; s>>=1) { if (t<s) red[t]+=red[t+s]; __syncthreads(); }
  if (t==0) atomicAdd(goodness, red[0]/tau - lse);
}

void scff_signal_launch(at::Tensor z, at::Tensor zpos, at::Tensor s_perp,
                        at::Tensor goodness, double tau, int64_t B, int64_t C) {
  const int threads = 128;
  const size_t shmem = (3*C + B + threads) * sizeof(float);
  scff_signal_kernel<<<B, threads, shmem>>>(
      z.data_ptr<float>(), zpos.data_ptr<float>(), s_perp.data_ptr<float>(),
      goodness.data_ptr<float>(), (float)tau, (int)B, (int)C);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scff_signal", &scff_signal_launch); }

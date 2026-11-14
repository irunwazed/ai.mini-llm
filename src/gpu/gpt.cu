/*
 * BACKEND GPU (CUDA) UNTUK TRAINING GPT — ditulis sendiri, TANPA cuBLAS/cuDNN/library ML.
 *
 * Isinya sama persis dengan versi CPU (src/gpt/layers.ts + src/gpt/model.ts), hanya dijalankan
 * ribuan thread GPU sekaligus dan memakai float32:
 *   - semua baris (token) dari B urutan diproses bersama: N = B·T baris
 *   - matmul memakai kernel "tiled" (potongan 64×64 dimuat ke shared memory, tiap thread menghitung 4×4)
 *   - Layer Norm: 1 warp (32 thread) per baris, jumlah dihitung dengan __shfl_xor_sync
 *   - attention causal: 1 blok per (urutan·head, posisi i)
 *   - parameter, gradien, momentum Adam, dan semua aktivasi TINGGAL di memori GPU;
 *     per langkah hanya token yang dikirim dan 1 angka loss yang diambil
 *
 * Dipanggil dari Bun lewat bun:ffi (src/gpu/cuda.ts). Satu model per proses.
 * Kompilasi:  bun run build-gpu   (nvcc -O3 -arch=native -shared -Xcompiler -fPIC)
 * Uji:        bun run test-gpu    (bandingkan loss & gradien dengan versi CPU)
 */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ============================================================================ galat
static char g_galat[600] = "";
static bool galat(const char* apa, cudaError_t e, int baris) {
  snprintf(g_galat, sizeof g_galat, "%s: %s (gpt.cu:%d)", apa, cudaGetErrorString(e), baris);
  return false;
}
#define CEK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return galat(#x, e_, __LINE__); } while (0)
#define SELESAI() do { cudaError_t e_ = cudaGetLastError(); if (e_ == cudaSuccess) e_ = cudaDeviceSynchronize(); if (e_ != cudaSuccess) { galat("kernel", e_, __LINE__); return -1; } } while (0)

static inline int blok(long n, int t = 256) { return (int)((n + t - 1) / t); }

// ============================================================================ reduksi
__device__ __forceinline__ float jumlahWarp(float x) {
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
  return x;
}
__device__ __forceinline__ float maksWarp(float x) {
  for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o));
  return x;
}
// jumlah/maks seluruh blok (blockDim kelipatan 32, maks 1024); hasil sama di semua thread
__device__ float jumlahBlok(float x, float* sh) {
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  x = jumlahWarp(x);
  __syncthreads();
  if (lane == 0) sh[w] = x;
  __syncthreads();
  float t = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
  if (w == 0) t = jumlahWarp(t);
  if (threadIdx.x == 0) sh[0] = t;
  __syncthreads();
  return sh[0];
}
__device__ float maksBlok(float x, float* sh) {
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  x = maksWarp(x);
  __syncthreads();
  if (lane == 0) sh[w] = x;
  __syncthreads();
  float t = threadIdx.x < nw ? sh[threadIdx.x] : -INFINITY;
  if (w == 0) t = maksWarp(t);
  if (threadIdx.x == 0) sh[0] = t;
  __syncthreads();
  return sh[0];
}

// ============================================================================ matmul (tiled)
// C(M×N) = op(A)·op(B) + beta·C      op(A)(m,k) = ta ? A[k·lda+m] : A[m·lda+k]
//                                    op(B)(k,n) = tb ? B[n·ldb+k] : B[k·ldb+n]
// Setiap blok menghitung potongan 64×64 dari C; 256 thread × (4×4) elemen.
#define TB 64
#define TK 16
__global__ void k_gemm(const float* __restrict__ A, const float* __restrict__ B, float* C,
                       int M, int N, int K, int lda, int ldb, int ta, int tb, float beta) {
  __shared__ float As[TK][TB + 1];
  __shared__ float Bs[TK][TB + 1];
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int m0 = blockIdx.y * TB, n0 = blockIdx.x * TB;
  float acc[4][4] = {{0}};
  for (int k0 = 0; k0 < K; k0 += TK) {
    for (int e = threadIdx.x; e < TB * TK; e += 256) {
      int mm, kk;
      if (ta) { kk = e / TB; mm = e % TB; } else { mm = e / TK; kk = e % TK; } // baca berurutan di memori
      int gm = m0 + mm, gk = k0 + kk;
      float v = 0.f;
      if (gm < M && gk < K) v = ta ? A[(long)gk * lda + gm] : A[(long)gm * lda + gk];
      As[kk][mm] = v;
    }
    for (int e = threadIdx.x; e < TB * TK; e += 256) {
      int kk, nn;
      if (tb) { nn = e / TK; kk = e % TK; } else { kk = e / TB; nn = e % TB; }
      int gk = k0 + kk, gn = n0 + nn;
      float v = 0.f;
      if (gk < K && gn < N) v = tb ? B[(long)gn * ldb + gk] : B[(long)gk * ldb + gn];
      Bs[kk][nn] = v;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TK; kk++) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; i++) a[i] = As[kk][ty + 16 * i];
#pragma unroll
      for (int j = 0; j < 4; j++) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int gm = m0 + ty + 16 * i;
    if (gm >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int gn = n0 + tx + 16 * j;
      if (gn >= N) continue;
      long idx = (long)gm * N + gn;
      C[idx] = acc[i][j] + (beta != 0.f ? beta * C[idx] : 0.f);
    }
  }
}
static void gemm(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, bool ta, bool tb, float beta) {
  dim3 grid((N + TB - 1) / TB, (M + TB - 1) / TB);
  k_gemm<<<grid, 256>>>(A, B, C, M, N, K, lda, ldb, ta, tb, beta);
}

// ============================================================================ elementwise
__global__ void k_isi_bias(float* Y, const float* b, long n, int O) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) Y[i] = b[i % O]; }
__global__ void k_salin(float* Y, const float* X, long n) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) Y[i] = X[i]; }
__global__ void k_jumlah(float* Y, const float* A, const float* B, long n) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) Y[i] = A[i] + B[i]; }
__global__ void k_tambah(float* Y, const float* X, long n) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) Y[i] += X[i]; }
__global__ void k_kali(float* Y, const float* M, long n) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) Y[i] *= M[i]; }
__global__ void k_nol(float* Y, long n) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) Y[i] = 0.f; }

#define SQRT_2_PI 0.7978845608028654f
__global__ void k_gelu(const float* X, float* Y, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) { float x = X[i]; Y[i] = 0.5f * x * (1.f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x))); }
}
__global__ void k_gelu_mundur(const float* X, const float* dY, float* dX, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) {
    float x = X[i], th = tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x));
    float turunan = 0.5f * (1.f + th) + 0.5f * x * (1.f - th * th) * SQRT_2_PI * (1.f + 3.f * 0.044715f * x * x);
    dX[i] += dY[i] * turunan;
  }
}

// Dropout: bilangan acak dari hash (indeks, seed) -> tidak perlu menyimpan state RNG
__device__ __forceinline__ float acak01(unsigned long long i, unsigned long long seed) {
  unsigned long long z = i * 0x9E3779B97F4A7C15ULL + seed * 0xD1B54A32D192ED03ULL + 0x632BE59BD9B4E019ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return (float)(z >> 40) * (1.0f / 16777216.0f);
}
__global__ void k_mask(float* M, long n, float p, unsigned long long seed) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) M[i] = acak01(i, seed) < p ? 0.f : 1.f / (1.f - p);
}

// ============================================================================ embedding
// x0[n,d] = √D·E[id_n, d] + PE[t, d]       (t = n mod T)
__global__ void k_embed(const int* ids, const float* E, const float* pe, float* X, long nD, int T, int D, float sD) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= nD) return;
  long n = i / D; int d = i % D, t = n % T;
  X[i] = sD * E[(long)ids[n] * D + d] + pe[(long)t * D + d];
}
// ∂L/∂E[id] += √D·∂L/∂x0
__global__ void k_embed_mundur(const int* ids, const float* dX, float* GE, long nD, int D, float sD) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= nD) return;
  long n = i / D; int d = i % D;
  atomicAdd(&GE[(long)ids[n] * D + d], sD * dX[i]);
}

// ============================================================================ Layer Norm
// 1 warp per baris: μ, σ², x̂ = (x−μ)·r, y = γ·x̂ + β
__global__ void k_ln(const float* X, int N, int D, const float* g, const float* b, float* Y, float* Xh, float* rstd) {
  int baris = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (baris >= N) return;
  const float* x = X + (long)baris * D;
  float s = 0.f;
  for (int d = lane; d < D; d += 32) s += x[d];
  float mu = jumlahWarp(s) / D;
  float v = 0.f;
  for (int d = lane; d < D; d += 32) { float t = x[d] - mu; v += t * t; }
  float r = rsqrtf(jumlahWarp(v) / D + 1e-5f);
  if (lane == 0) rstd[baris] = r;
  for (int d = lane; d < D; d += 32) {
    float xh = (x[d] - mu) * r;
    Xh[(long)baris * D + d] = xh;
    Y[(long)baris * D + d] = g[d] * xh + b[d];
  }
}
// ∂L/∂x += (r/D)·(D·g − Σg − x̂·Σ(g·x̂)),   g = dy·γ
__global__ void k_ln_mundur(const float* dY, int N, int D, const float* gam, const float* Xh, const float* rstd, float* dX) {
  int baris = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (baris >= N) return;
  long o = (long)baris * D;
  float sg = 0.f, sgx = 0.f;
  for (int d = lane; d < D; d += 32) { float g = dY[o + d] * gam[d]; sg += g; sgx += g * Xh[o + d]; }
  sg = jumlahWarp(sg); sgx = jumlahWarp(sgx);
  float r = rstd[baris] / D;
  for (int d = lane; d < D; d += 32) { float g = dY[o + d] * gam[d]; dX[o + d] += r * (D * g - sg - Xh[o + d] * sgx); }
}
// Jumlah per kolom (untuk ∂γ = Σ dy·x̂, ∂β = Σ dy, ∂b = Σ dy): blok mengurus 128 kolom × 256 baris, lalu atomicAdd
#define BARIS_KOLOM 256
__global__ void k_kolom(const float* dY, const float* Xh, int N, int O, float* Gg, float* Gb) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= O) return;
  int r0 = blockIdx.y * BARIS_KOLOM, r1 = min(N, r0 + BARIS_KOLOM);
  float sg = 0.f, sb = 0.f;
  for (int r = r0; r < r1; r++) {
    float dy = dY[(long)r * O + c];
    sb += dy;
    if (Xh) sg += dy * Xh[(long)r * O + c];
  }
  if (Gg) atomicAdd(&Gg[c], sg);
  atomicAdd(&Gb[c], sb);
}
static void kolom(const float* dY, const float* Xh, int N, int O, float* Gg, float* Gb) {
  dim3 grid((O + 127) / 128, (N + BARIS_KOLOM - 1) / BARIS_KOLOM);
  k_kolom<<<grid, 128>>>(dY, Xh, N, O, Gg, Gb);
}

// ============================================================================ attention causal
// blok = (posisi i, urutan·head bh); P disimpan [bh][i][j] (T×T per bh)
__global__ void k_attn(const float* Q, const float* K, const float* V, float* Pa, float* O, int T, int D, int H, int dh, float skala) {
  extern __shared__ float sh[];
  float* skor = sh;            // T
  float* qi = sh + T;          // dh
  float* red = sh + T + dh;    // 32
  const int i = blockIdx.x, bh = blockIdx.y, b = bh / H, h = bh % H, c = h * dh;
  const float* q = Q + ((long)b * T + i) * D + c;
  for (int d = threadIdx.x; d < dh; d += blockDim.x) qi[d] = q[d];
  __syncthreads();
  float m = -INFINITY;
  for (int j = threadIdx.x; j <= i; j += blockDim.x) {
    const float* k = K + ((long)b * T + j) * D + c;
    float s = 0.f;
    for (int d = 0; d < dh; d++) s += qi[d] * k[d];
    s *= skala;
    skor[j] = s;
    m = fmaxf(m, s);
  }
  m = maksBlok(m, red);
  float jum = 0.f;
  for (int j = threadIdx.x; j <= i; j += blockDim.x) { float e = expf(skor[j] - m); skor[j] = e; jum += e; }
  jum = jumlahBlok(jum, red);
  float* p = Pa + ((long)bh * T + i) * T;
  for (int j = threadIdx.x; j < T; j += blockDim.x) {
    if (j <= i) { float v = skor[j] / jum; skor[j] = v; p[j] = v; } else p[j] = 0.f; // masa depan = 0
  }
  __syncthreads();
  for (int d = threadIdx.x; d < dh; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j <= i; j++) acc += skor[j] * V[((long)b * T + j) * D + c + d];
    O[((long)b * T + i) * D + c + d] = acc;
  }
}
// dP_ij = dO_i·V_j ;  titik = Σ_j P_ij dP_ij ;  dS_ij = P_ij (dP_ij − titik)/√dh
__global__ void k_attn_ds(const float* V, const float* Pa, const float* dO, float* dS, int T, int D, int H, int dh, float skala) {
  extern __shared__ float sh[];
  float* dp = sh; float* doi = sh + T; float* red = sh + T + dh;
  const int i = blockIdx.x, bh = blockIdx.y, b = bh / H, h = bh % H, c = h * dh;
  for (int d = threadIdx.x; d < dh; d += blockDim.x) doi[d] = dO[((long)b * T + i) * D + c + d];
  __syncthreads();
  const float* p = Pa + ((long)bh * T + i) * T;
  float titik = 0.f;
  for (int j = threadIdx.x; j <= i; j += blockDim.x) {
    const float* v = V + ((long)b * T + j) * D + c;
    float s = 0.f;
    for (int d = 0; d < dh; d++) s += doi[d] * v[d];
    dp[j] = s;
    titik += p[j] * s;
  }
  titik = jumlahBlok(titik, red);
  float* ds = dS + ((long)bh * T + i) * T;
  for (int j = threadIdx.x; j < T; j += blockDim.x) ds[j] = j <= i ? p[j] * (dp[j] - titik) * skala : 0.f;
}
// dV_j += Σ_{i≥j} P_ij dO_i          (blok per (j, bh), thread per dimensi)
__global__ void k_attn_dv(const float* Pa, const float* dO, float* dV, int T, int D, int H, int dh) {
  const int j = blockIdx.x, bh = blockIdx.y, b = bh / H, c = (bh % H) * dh;
  for (int d = threadIdx.x; d < dh; d += blockDim.x) {
    float acc = 0.f;
    for (int i = j; i < T; i++) acc += Pa[((long)bh * T + i) * T + j] * dO[((long)b * T + i) * D + c + d];
    dV[((long)b * T + j) * D + c + d] += acc;
  }
}
// dQ_i += Σ_{j≤i} dS_ij K_j
__global__ void k_attn_dq(const float* dS, const float* K, float* dQ, int T, int D, int H, int dh) {
  const int i = blockIdx.x, bh = blockIdx.y, b = bh / H, c = (bh % H) * dh;
  const float* ds = dS + ((long)bh * T + i) * T;
  for (int d = threadIdx.x; d < dh; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j <= i; j++) acc += ds[j] * K[((long)b * T + j) * D + c + d];
    dQ[((long)b * T + i) * D + c + d] += acc;
  }
}
// dK_j += Σ_{i≥j} dS_ij Q_i
__global__ void k_attn_dk(const float* dS, const float* Q, float* dK, int T, int D, int H, int dh) {
  const int j = blockIdx.x, bh = blockIdx.y, b = bh / H, c = (bh % H) * dh;
  for (int d = threadIdx.x; d < dh; d += blockDim.x) {
    float acc = 0.f;
    for (int i = j; i < T; i++) acc += dS[((long)bh * T + i) * T + j] * Q[((long)b * T + i) * D + c + d];
    dK[((long)b * T + j) * D + c + d] += acc;
  }
}

// ============================================================================ softmax + cross-entropy
// blok per baris; target −1 = diabaikan. Loss dijumlah (double) ke *loss.
__global__ void k_ce(const float* Z, const int* tgt, int V, float* dZ, float skala, double* loss) {
  __shared__ float red[32];
  const int baris = blockIdx.x, t = tgt[baris];
  const float* z = Z + (long)baris * V;
  if (t < 0) {
    if (dZ) for (int v = threadIdx.x; v < V; v += blockDim.x) dZ[(long)baris * V + v] = 0.f;
    return;
  }
  float m = -INFINITY;
  for (int v = threadIdx.x; v < V; v += blockDim.x) m = fmaxf(m, z[v]);
  m = maksBlok(m, red);
  float s = 0.f;
  for (int v = threadIdx.x; v < V; v += blockDim.x) s += expf(z[v] - m);
  s = jumlahBlok(s, red);
  const float logJ = logf(s) + m;
  if (threadIdx.x == 0) atomicAdd(loss, (double)(logJ - z[t]));
  if (dZ) for (int v = threadIdx.x; v < V; v += blockDim.x) dZ[(long)baris * V + v] = expf(z[v] - logJ) * skala - (v == t ? skala : 0.f);
}

// ============================================================================ AdamW
__global__ void k_kuadrat(const float* G, long n, double* out) {
  __shared__ float red[32];
  float s = 0.f;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) s += G[i] * G[i];
  s = jumlahBlok(s, red);
  if (threadIdx.x == 0) atomicAdd(out, (double)s);
}
__global__ void k_adam(float* P, const float* G, float* m, float* v, const unsigned char* decay, long n,
                       float lr, float b1, float b2, float eps, float wd, float clip, float c1, float c2, const double* kuadrat) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  float norma = (float)sqrt(*kuadrat);
  float k = (clip > 0.f && norma > clip) ? clip / norma : 1.f;
  float g = G[i] * k;
  float mi = b1 * m[i] + (1.f - b1) * g, vi = b2 * v[i] + (1.f - b2) * g * g;
  m[i] = mi; v[i] = vi;
  float u = mi / c1 / (sqrtf(vi / c2) + eps);
  if (decay[i]) u += wd * P[i];
  P[i] -= lr * u;
}

// ============================================================================ model
struct Off { long wq, bq, wk, bk, wv, bv, wo, bo, ln1g, ln1b, w1, b1, w2, b2, ln2g, ln2b; };
struct Lapisan {
  float *x, *u, *q, *k, *v, *p, *o, *a, *s1, *h, *w, *f1, *g, *f2, *s2, *y, *ln1hat, *ln1r, *ln2hat, *ln2r, *ma, *mf;
};
struct Model {
  int V, C, D, H, L, F, preLN, Bmaks;
  long ukuran, tok, lnfg, lnfb, wout, bout;
  std::vector<Off> off;
  std::vector<Lapisan> lap;
  float *P, *G, *m, *v; unsigned char* decay;
  float* pe; int *ids, *tgt;
  float *x0, *m0, *akhir, *lnfhat, *lnfr, *logits, *dLogits;
  float *dY, *dH, *dS, *dU, *dO, *dA, *dQ, *dK, *dV, *dG, *dF1, *dSatt;
  double* angka; // [0] = loss, [1] = Σg²
  std::vector<void*> semua;
};
static Model* M = nullptr;

static bool alokasi(float** p, long n) {
  CEK(cudaMalloc((void**)p, sizeof(float) * (n > 0 ? n : 1)));
  M->semua.push_back(*p);
  return true;
}

// ---- operasi gabungan (sama dengan versi CPU)
static void linear(const float* X, int N, int I, int O, long wOff, long bOff, float* Y) {
  long n = (long)N * O;
  k_isi_bias<<<blok(n), 256>>>(Y, M->P + bOff, n, O);
  gemm(X, M->P + wOff, Y, N, O, I, I, O, false, false, 1.f);
}
// ∂W += Xᵀ·dY,  ∂b += Σ dY,  dX += dY·Wᵀ (jika dX)
static void linearMundur(const float* X, const float* dY, int N, int I, int O, long wOff, long bOff, float* dX) {
  gemm(X, dY, M->G + wOff, I, O, N, I, O, true, false, 1.f);
  kolom(dY, nullptr, N, O, nullptr, M->G + bOff);
  if (dX) gemm(dY, M->P + wOff, dX, N, I, O, O, O, false, true, 1.f);
}
static void ln(const float* X, int N, long gOff, long bOff, float* Y, float* Xh, float* r) {
  k_ln<<<blok(N, 8), 256>>>(X, N, M->D, M->P + gOff, M->P + bOff, Y, Xh, r);
}
static void lnMundur(const float* dY, int N, long gOff, long bOff, const float* Xh, const float* r, float* dX) {
  kolom(dY, Xh, N, M->D, M->G + gOff, M->G + bOff);
  k_ln_mundur<<<blok(N, 8), 256>>>(dY, N, M->D, M->P + gOff, Xh, r, dX);
}
static void salin(float* Y, const float* X, long n) { k_salin<<<blok(n), 256>>>(Y, X, n); }
static void nol(float* Y, long n) { k_nol<<<blok(n), 256>>>(Y, n); }
static void mask(float* Mk, long n, float p, unsigned long long seed) { k_mask<<<blok(n), 256>>>(Mk, n, p, seed); }
static void kali(float* Y, const float* Mk, long n) { k_kali<<<blok(n), 256>>>(Y, Mk, n); }

static int benangAttn(int T) { int t = 32; while (t < T && t < 256) t <<= 1; return t; }

static void maju(int B, int T, float p, unsigned long long seed) {
  const int D = M->D, H = M->H, F = M->F, V = M->V, N = B * T, dh = D / H;
  const long n = (long)N * D;
  const bool drop = p > 0.f;
  const float skala = 1.f / sqrtf((float)dh);
  k_embed<<<blok(n), 256>>>(M->ids, M->P + M->tok, M->pe, M->x0, n, T, D, sqrtf((float)D));
  if (drop) { mask(M->m0, n, p, seed * 7919ULL + 1); kali(M->x0, M->m0, n); }
  float* x = M->x0;
  const int bt = benangAttn(T);
  const size_t shAttn = sizeof(float) * (T + dh + 32);
  for (int l = 0; l < M->L; l++) {
    Lapisan& Lp = M->lap[l]; const Off& o = M->off[l];
    salin(Lp.x, x, n);
    if (M->preLN) ln(Lp.x, N, o.ln1g, o.ln1b, Lp.u, Lp.ln1hat, Lp.ln1r); else salin(Lp.u, Lp.x, n);
    linear(Lp.u, N, D, D, o.wq, o.bq, Lp.q);
    linear(Lp.u, N, D, D, o.wk, o.bk, Lp.k);
    linear(Lp.u, N, D, D, o.wv, o.bv, Lp.v);
    k_attn<<<dim3(T, B * H), bt, shAttn>>>(Lp.q, Lp.k, Lp.v, Lp.p, Lp.o, T, D, H, dh, skala);
    linear(Lp.o, N, D, D, o.wo, o.bo, Lp.a);
    if (drop) { mask(Lp.ma, n, p, seed * 7919ULL + 2 + 2 * l); kali(Lp.a, Lp.ma, n); }
    k_jumlah<<<blok(n), 256>>>(Lp.s1, Lp.x, Lp.a, n);
    if (M->preLN) { salin(Lp.h, Lp.s1, n); ln(Lp.h, N, o.ln2g, o.ln2b, Lp.w, Lp.ln2hat, Lp.ln2r); }
    else { ln(Lp.s1, N, o.ln1g, o.ln1b, Lp.h, Lp.ln1hat, Lp.ln1r); salin(Lp.w, Lp.h, n); }
    linear(Lp.w, N, D, F, o.w1, o.b1, Lp.f1);
    k_gelu<<<blok((long)N * F), 256>>>(Lp.f1, Lp.g, (long)N * F);
    linear(Lp.g, N, F, D, o.w2, o.b2, Lp.f2);
    if (drop) { mask(Lp.mf, n, p, seed * 7919ULL + 3 + 2 * l); kali(Lp.f2, Lp.mf, n); }
    if (M->preLN) k_jumlah<<<blok(n), 256>>>(Lp.y, Lp.h, Lp.f2, n);
    else { k_jumlah<<<blok(n), 256>>>(Lp.s2, Lp.h, Lp.f2, n); ln(Lp.s2, N, o.ln2g, o.ln2b, Lp.y, Lp.ln2hat, Lp.ln2r); }
    x = Lp.y;
  }
  if (M->preLN) ln(x, N, M->lnfg, M->lnfb, M->akhir, M->lnfhat, M->lnfr); else salin(M->akhir, x, n);
  linear(M->akhir, N, D, V, M->wout, M->bout, M->logits);
}

static void mundur(int B, int T, bool drop) {
  const int D = M->D, H = M->H, F = M->F, V = M->V, N = B * T, dh = D / H;
  const long n = (long)N * D, nF = (long)N * F;
  const float skala = 1.f / sqrtf((float)dh);
  const int bt = benangAttn(T);
  const int bd = dh <= 32 ? 32 : 64;
  nol(M->dY, n);
  linearMundur(M->akhir, M->dLogits, N, D, V, M->wout, M->bout, M->dY);
  if (M->preLN) { nol(M->dS, n); lnMundur(M->dY, N, M->lnfg, M->lnfb, M->lnfhat, M->lnfr, M->dS); salin(M->dY, M->dS, n); }
  for (int l = M->L - 1; l >= 0; l--) {
    Lapisan& Lp = M->lap[l]; const Off& o = M->off[l];
    nol(M->dS, n);
    if (M->preLN) salin(M->dS, M->dY, n); else lnMundur(M->dY, N, o.ln2g, o.ln2b, Lp.ln2hat, Lp.ln2r, M->dS);
    salin(M->dH, M->dS, n);
    if (drop) kali(M->dS, Lp.mf, n);
    nol(M->dG, nF);
    linearMundur(Lp.g, M->dS, N, F, D, o.w2, o.b2, M->dG);
    nol(M->dF1, nF);
    k_gelu_mundur<<<blok(nF), 256>>>(Lp.f1, M->dG, M->dF1, nF);
    nol(M->dU, n);
    linearMundur(Lp.w, M->dF1, N, D, F, o.w1, o.b1, M->dU);
    if (M->preLN) lnMundur(M->dU, N, o.ln2g, o.ln2b, Lp.ln2hat, Lp.ln2r, M->dH);
    else k_tambah<<<blok(n), 256>>>(M->dH, M->dU, n);
    nol(M->dA, n);
    if (M->preLN) salin(M->dA, M->dH, n); else lnMundur(M->dH, N, o.ln1g, o.ln1b, Lp.ln1hat, Lp.ln1r, M->dA);
    salin(M->dY, M->dA, n);
    if (drop) kali(M->dA, Lp.ma, n);
    nol(M->dO, n);
    linearMundur(Lp.o, M->dA, N, D, D, o.wo, o.bo, M->dO);
    nol(M->dQ, n); nol(M->dK, n); nol(M->dV, n);
    k_attn_ds<<<dim3(T, B * H), bt, sizeof(float) * (T + dh + 32)>>>(Lp.v, Lp.p, M->dO, M->dSatt, T, D, H, dh, skala);
    k_attn_dv<<<dim3(T, B * H), bd>>>(Lp.p, M->dO, M->dV, T, D, H, dh);
    k_attn_dq<<<dim3(T, B * H), bd>>>(M->dSatt, Lp.k, M->dQ, T, D, H, dh);
    k_attn_dk<<<dim3(T, B * H), bd>>>(M->dSatt, Lp.q, M->dK, T, D, H, dh);
    nol(M->dU, n);
    linearMundur(Lp.u, M->dQ, N, D, D, o.wq, o.bq, M->dU);
    linearMundur(Lp.u, M->dK, N, D, D, o.wk, o.bk, M->dU);
    linearMundur(Lp.u, M->dV, N, D, D, o.wv, o.bv, M->dU);
    if (M->preLN) lnMundur(M->dU, N, o.ln1g, o.ln1b, Lp.ln1hat, Lp.ln1r, M->dY);
    else k_tambah<<<blok(n), 256>>>(M->dY, M->dU, n);
  }
  if (drop) kali(M->dY, M->m0, n);
  k_embed_mundur<<<blok(n), 256>>>(M->ids, M->dY, M->G + M->tok, n, D, sqrtf((float)D));
}

// ============================================================================ API C (dipanggil dari Bun)
extern "C" {

// cfg = [V, C, D, H, L, F, preLN, Bmaks]
// off = [tok, per lapisan 16 offset (wq bq wk bk wv bv wo bo ln1g ln1b w1 b1 w2 b2 ln2g ln2b), lnfg, lnfb, wout, bout]
static int buatDalam(const int* cfg, const int* off, int nOff, int ukuran, const unsigned char* decay) {
  if (M) { snprintf(g_galat, sizeof g_galat, "model GPU sudah dibuat (satu per proses)"); return -1; }
  int nDev = 0;
  if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev == 0) { snprintf(g_galat, sizeof g_galat, "tidak ada GPU CUDA"); return -1; }
  M = new Model();
  M->V = cfg[0]; M->C = cfg[1]; M->D = cfg[2]; M->H = cfg[3]; M->L = cfg[4]; M->F = cfg[5]; M->preLN = cfg[6]; M->Bmaks = cfg[7];
  if (nOff != 1 + 16 * M->L + 4) { snprintf(g_galat, sizeof g_galat, "jumlah offset %d tidak cocok", nOff); return -1; }
  if (M->D % M->H != 0 || M->C > 4096) { snprintf(g_galat, sizeof g_galat, "konfigurasi tidak didukung"); return -1; }
  M->ukuran = ukuran; M->tok = off[0];
  for (int l = 0; l < M->L; l++) {
    const int* a = off + 1 + 16 * l;
    M->off.push_back({a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]});
  }
  const int* e = off + 1 + 16 * M->L;
  M->lnfg = e[0]; M->lnfb = e[1]; M->wout = e[2]; M->bout = e[3];
  const long N = (long)M->Bmaks * M->C, D = M->D, F = M->F, BHTT = (long)M->Bmaks * M->H * M->C * M->C;
  auto f = [&](float** p, long n) { return alokasi(p, n); };
  if (!f(&M->P, ukuran) || !f(&M->G, ukuran) || !f(&M->m, ukuran) || !f(&M->v, ukuran)) return -1;
  if (cudaMalloc((void**)&M->decay, ukuran) != cudaSuccess) { snprintf(g_galat, sizeof g_galat, "memori GPU tidak cukup"); return -1; }
  M->semua.push_back(M->decay);
  if (cudaMemcpy(M->decay, decay, ukuran, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
  cudaMemset(M->m, 0, sizeof(float) * ukuran); cudaMemset(M->v, 0, sizeof(float) * ukuran); cudaMemset(M->G, 0, sizeof(float) * ukuran);
  // positional encoding sinusoidal (sama dengan positionalEncoding di layers.ts)
  std::vector<float> pe((size_t)M->C * D, 0.f);
  for (int pos = 0; pos < M->C; pos++)
    for (int i = 0; i < D; i += 2) {
      double sudut = pos / pow(10000.0, (double)i / D);
      pe[(size_t)pos * D + i] = (float)sin(sudut);
      if (i + 1 < D) pe[(size_t)pos * D + i + 1] = (float)cos(sudut);
    }
  if (!f(&M->pe, (long)M->C * D)) return -1;
  cudaMemcpy(M->pe, pe.data(), sizeof(float) * pe.size(), cudaMemcpyHostToDevice);
  if (cudaMalloc((void**)&M->ids, sizeof(int) * N) != cudaSuccess || cudaMalloc((void**)&M->tgt, sizeof(int) * N) != cudaSuccess) { snprintf(g_galat, sizeof g_galat, "memori GPU tidak cukup"); return -1; }
  M->semua.push_back(M->ids); M->semua.push_back(M->tgt);
  if (cudaMalloc((void**)&M->angka, sizeof(double) * 2) != cudaSuccess) return -1;
  M->semua.push_back(M->angka);
  bool ok = f(&M->x0, N * D) && f(&M->m0, N * D) && f(&M->akhir, N * D) && f(&M->lnfhat, N * D) && f(&M->lnfr, N)
    && f(&M->logits, N * M->V) && f(&M->dLogits, N * M->V)
    && f(&M->dY, N * D) && f(&M->dH, N * D) && f(&M->dS, N * D) && f(&M->dU, N * D) && f(&M->dO, N * D) && f(&M->dA, N * D)
    && f(&M->dQ, N * D) && f(&M->dK, N * D) && f(&M->dV, N * D) && f(&M->dG, N * F) && f(&M->dF1, N * F) && f(&M->dSatt, BHTT);
  if (!ok) { snprintf(g_galat, sizeof g_galat, "memori GPU tidak cukup untuk batch %d × konteks %d (kecilkan --batch)", M->Bmaks, M->C); return -1; }
  for (int l = 0; l < M->L; l++) {
    Lapisan Lp;
    ok = f(&Lp.x, N * D) && f(&Lp.u, N * D) && f(&Lp.q, N * D) && f(&Lp.k, N * D) && f(&Lp.v, N * D) && f(&Lp.p, BHTT)
      && f(&Lp.o, N * D) && f(&Lp.a, N * D) && f(&Lp.s1, N * D) && f(&Lp.h, N * D) && f(&Lp.w, N * D)
      && f(&Lp.f1, N * F) && f(&Lp.g, N * F) && f(&Lp.f2, N * D) && f(&Lp.s2, N * D) && f(&Lp.y, N * D)
      && f(&Lp.ln1hat, N * D) && f(&Lp.ln1r, N) && f(&Lp.ln2hat, N * D) && f(&Lp.ln2r, N) && f(&Lp.ma, N * D) && f(&Lp.mf, N * D);
    if (!ok) { snprintf(g_galat, sizeof g_galat, "memori GPU tidak cukup untuk batch %d × konteks %d (kecilkan --batch)", M->Bmaks, M->C); return -1; }
    M->lap.push_back(Lp);
  }
  return 0;
}

int gc_hapus();
// Buat model di GPU; jika gagal di tengah (mis. memori tidak cukup), semua yang sudah dialokasi dibebaskan.
int gc_buat(const int* cfg, const int* off, int nOff, int ukuran, const unsigned char* decay) {
  if (M) { snprintf(g_galat, sizeof g_galat, "model GPU sudah dibuat (satu per proses)"); return -1; }
  int r = buatDalam(cfg, off, nOff, ukuran, decay);
  if (r != 0 && M) { char simpan[600]; memcpy(simpan, g_galat, sizeof simpan); gc_hapus(); memcpy(g_galat, simpan, sizeof simpan); }
  return r;
}

static bool f64keGPU(float* tujuan, const double* sumber, long n) {
  std::vector<float> t(n);
  for (long i = 0; i < n; i++) t[i] = (float)sumber[i];
  CEK(cudaMemcpy(tujuan, t.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  return true;
}
static bool gpuKeF64(double* tujuan, const float* sumber, long n) {
  std::vector<float> t(n);
  CEK(cudaMemcpy(t.data(), sumber, sizeof(float) * n, cudaMemcpyDeviceToHost));
  for (long i = 0; i < n; i++) tujuan[i] = t[i];
  return true;
}

int gc_isi_param(const double* P) { return M && f64keGPU(M->P, P, M->ukuran) ? 0 : -1; }
int gc_ambil_param(double* P) { return M && gpuKeF64(P, M->P, M->ukuran) ? 0 : -1; }
int gc_ambil_grad(double* G) { return M && gpuKeF64(G, M->G, M->ukuran) ? 0 : -1; }
int gc_isi_adam(const double* m, const double* v) { return M && f64keGPU(M->m, m, M->ukuran) && f64keGPU(M->v, v, M->ukuran) ? 0 : -1; }
int gc_ambil_adam(double* m, double* v) { return M && gpuKeF64(m, M->m, M->ukuran) && gpuKeF64(v, M->v, M->ukuran) ? 0 : -1; }

int gc_nol_grad() {
  if (!M) return -1;
  nol(M->G, M->ukuran);
  SELESAI();
  return 0;
}

static int siapkanBatch(const int* ids, const int* tgt, int B, int T) {
  if (!M) { snprintf(g_galat, sizeof g_galat, "model GPU belum dibuat"); return -1; }
  if (B < 1 || B > M->Bmaks || T < 1 || T > M->C) { snprintf(g_galat, sizeof g_galat, "batch %d × %d melebihi batas %d × %d", B, T, M->Bmaks, M->C); return -1; }
  if (cudaMemcpy(M->ids, ids, sizeof(int) * B * T, cudaMemcpyHostToDevice) != cudaSuccess
   || cudaMemcpy(M->tgt, tgt, sizeof(int) * B * T, cudaMemcpyHostToDevice) != cudaSuccess) { snprintf(g_galat, sizeof g_galat, "gagal menyalin token ke GPU"); return -1; }
  cudaMemset(M->angka, 0, sizeof(double) * 2);
  return 0;
}

// Forward + loss + backward untuk B urutan sepanjang T (padding: target −1). Gradien DITAMBAHKAN ke G.
// *loss = jumlah −log p(target) (belum dirata-rata), sama seperti GPT.langkah di CPU.
int gc_langkah(const int* ids, const int* tgt, int B, int T, float skala, float dropout, int seed, double* loss) {
  if (siapkanBatch(ids, tgt, B, T)) return -1;
  maju(B, T, dropout, (unsigned long long)(unsigned)seed + 1ULL);
  k_ce<<<B * T, 256>>>(M->logits, M->tgt, M->V, M->dLogits, skala, M->angka);
  mundur(B, T, dropout > 0.f);
  SELESAI();
  if (cudaMemcpy(loss, M->angka, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
  return 0;
}

// Hanya forward + loss (evaluasi, tanpa dropout)
int gc_loss(const int* ids, const int* tgt, int B, int T, double* loss) {
  if (siapkanBatch(ids, tgt, B, T)) return -1;
  maju(B, T, 0.f, 0);
  k_ce<<<B * T, 256>>>(M->logits, M->tgt, M->V, nullptr, 1.f, M->angka);
  SELESAI();
  if (cudaMemcpy(loss, M->angka, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
  return 0;
}

// Satu langkah AdamW (+ gradient clipping) memakai G di GPU. *norma = ‖g‖ sebelum clipping.
int gc_adam(int t, float lr, float b1, float b2, float eps, float wd, float clip, double* norma) {
  if (!M) return -1;
  cudaMemset(M->angka + 1, 0, sizeof(double));
  k_kuadrat<<<512, 256>>>(M->G, M->ukuran, M->angka + 1);
  const float c1 = (float)(1.0 - pow((double)b1, t)), c2 = (float)(1.0 - pow((double)b2, t));
  k_adam<<<blok(M->ukuran), 256>>>(M->P, M->G, M->m, M->v, M->decay, M->ukuran, lr, b1, b2, eps, wd, clip, c1, c2, M->angka + 1);
  SELESAI();
  double kuadrat = 0;
  if (cudaMemcpy(&kuadrat, M->angka + 1, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
  *norma = sqrt(kuadrat);
  return 0;
}

int gc_info(char* buf, int n) {
  int dev = 0; cudaDeviceProp p;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) { snprintf(buf, n, "tidak ada GPU"); return -1; }
  size_t bebas = 0, total = 0; cudaMemGetInfo(&bebas, &total);
  snprintf(buf, n, "%s · %.1f GB VRAM (%.1f GB bebas) · compute %d.%d · %d SM", p.name, total / 1e9, bebas / 1e9, p.major, p.minor, p.multiProcessorCount);
  return 0;
}

int gc_galat(char* buf, int n) { snprintf(buf, n, "%s", g_galat); return 0; }

int gc_hapus() {
  if (!M) return 0;
  for (void* p : M->semua) cudaFree(p);
  delete M; M = nullptr;
  return 0;
}

} // extern "C"

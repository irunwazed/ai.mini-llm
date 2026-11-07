/**
 * Operasi matriks dasar. Semua matriks disimpan row-major di Float64Array:
 *   elemen (i, j) dari matriks R×C berada di indeks i·C + j.
 *
 * Perkalian matriks C = A·B, dengan A (n×k), B (k×m), C (n×m):
 *   C[i][j] = Σ_t A[i][t] · B[t][j]
 *
 * Untuk backward kita butuh dua bentuk lain (turunan C = A·B):
 *   ∂L/∂A = ∂L/∂C · Bᵀ      -> matmulBT
 *   ∂L/∂B = Aᵀ · ∂L/∂C      -> matmulAT
 * Ketiganya MENAMBAHKAN ke keluaran (+=), karena gradien dari banyak jalur dijumlahkan.
 */
export type Vec = Float64Array;

/** out[n×m] += A[n×k] · B[k×m] */
export function matmul(A: Vec, B: Vec, out: Vec, n: number, k: number, m: number, aOff = 0, bOff = 0, oOff = 0): void {
  for (let i = 0; i < n; i++) {
    const ai = aOff + i * k, oi = oOff + i * m;
    for (let t = 0; t < k; t++) {
      const a = A[ai + t];
      if (a === 0) continue;
      const bt = bOff + t * m;
      for (let j = 0; j < m; j++) out[oi + j] += a * B[bt + j];
    }
  }
}

/** out[n×k] += dC[n×m] · Bᵀ, dengan B berukuran k×m */
export function matmulBT(dC: Vec, B: Vec, out: Vec, n: number, m: number, k: number, cOff = 0, bOff = 0, oOff = 0): void {
  for (let i = 0; i < n; i++) {
    const ci = cOff + i * m, oi = oOff + i * k;
    for (let j = 0; j < k; j++) {
      const bj = bOff + j * m;
      let s = 0;
      for (let t = 0; t < m; t++) s += dC[ci + t] * B[bj + t];
      out[oi + j] += s;
    }
  }
}

/** out[k×m] += Aᵀ · dC, dengan A berukuran n×k dan dC n×m */
export function matmulAT(A: Vec, dC: Vec, out: Vec, n: number, k: number, m: number, aOff = 0, cOff = 0, oOff = 0): void {
  for (let i = 0; i < n; i++) {
    const ai = aOff + i * k, ci = cOff + i * m;
    for (let t = 0; t < k; t++) {
      const a = A[ai + t];
      if (a === 0) continue;
      const ot = oOff + t * m;
      for (let j = 0; j < m; j++) out[ot + j] += a * dC[ci + j];
    }
  }
}

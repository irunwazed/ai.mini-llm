/**
 * Bilangan acak ber-seed, agar setiap training bisa diulang dengan hasil yang sama.
 *   mulberry32 : generator uniform [0, 1) yang cepat dan cukup baik untuk ML
 *   Box-Muller : dua bilangan uniform -> satu bilangan normal N(0, 1)
 *                z = sqrt(-2 ln u) · cos(2π v)
 */
export function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Rng {
  readonly next: () => number;
  constructor(seed = 42) {
    this.next = mulberry32(seed);
  }
  /** Normal N(0, 1) dengan Box-Muller. */
  normal(): number {
    const u = 1 - this.next();
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  /** Bilangan bulat acak 0..n-1. */
  int(n: number): number {
    return Math.floor(this.next() * n);
  }
}

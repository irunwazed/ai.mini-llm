/**
 * KOMPILASI BACKEND GPU: src/gpu/gpt.cu -> build/libgptcuda.so (hanya di mesin Linux ber-GPU NVIDIA + CUDA toolkit).
 *
 *   bun run build-gpu
 *
 * Memilih nvcc TERBARU yang terpasang (PATH, /usr/local/cuda*, /usr/local/cuda-X.Y). Arsitektur diambil dari
 * compute capability GPU; jika nvidia-smi tidak memberikannya (mis. GPU virtual), dikenali dari nama GPU
 * (RTX 50xx / RTX PRO 6000 Blackwell = sm_120). Blackwell butuh CUDA ≥ 12.8.
 * Setelah berhasil: bun run test-gpu (bandingkan dengan CPU), lalu train-gpt -- ... --gpu
 */
import { existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { FILE_LIB } from "../gpu/cuda.js";

const SUMBER = "src/gpu/gpt.cu";
const teks = async (cmd: string[]) => {
  try { const p = Bun.spawn(cmd, { stdout: "pipe", stderr: "pipe" }); const t = await new Response(p.stdout).text(); return (await p.exited) === 0 ? t.trim() : null; }
  catch { return null; }
};

if (process.platform !== "linux") {
  console.error("Backend GPU hanya dikompilasi di Linux dengan GPU NVIDIA (mis. server packet.ai). Di laptop ini training tetap memakai CPU.");
  process.exit(1);
}
const gpu = await teks(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"]);
if (!gpu) { console.error("nvidia-smi tidak menjawab: tidak ada GPU NVIDIA / driver belum terpasang."); process.exit(1); }
const [namaGPU = "", capTeks = ""] = gpu.split("\n")[0].split(",").map((x) => x.trim());
let cap = Number(capTeks);
if (!Number.isFinite(cap) || cap <= 0) cap = /RTX 50\d\d|RTX PRO \d+ Blackwell|Blackwell|B[12]00|GB[12]00/i.test(namaGPU) ? 12.0 : NaN;

// semua nvcc yang ada -> pilih versi tertinggi
const calon = new Set<string>();
const diPath = Bun.which("nvcc"); if (diPath) calon.add(diPath);
for (const d of existsSync("/usr/local") ? readdirSync("/usr/local") : []) if (/^cuda(-[\d.]+)?$/.test(d) && existsSync(`/usr/local/${d}/bin/nvcc`)) calon.add(`/usr/local/${d}/bin/nvcc`);
const versi: { nvcc: string; v: number[] }[] = [];
for (const n of calon) {
  const m = (await teks([n, "--version"]))?.match(/release (\d+)\.(\d+)/);
  if (m) versi.push({ nvcc: n, v: [Number(m[1]), Number(m[2])] });
}
versi.sort((a, b) => b.v[0] - a.v[0] || b.v[1] - a.v[1]);
if (!versi.length) { console.error("nvcc (CUDA toolkit) tidak ditemukan. Jalankan dari laptop: bun run server -- --siapkan (memasang otomatis)."); process.exit(1); }
const { nvcc, v: [mayor, minor] } = versi[0];
console.log(`GPU   : ${namaGPU} (compute ${Number.isFinite(cap) ? cap.toFixed(1) : "tidak diketahui"})`);
console.log(`nvcc  : ${nvcc} (CUDA ${mayor}.${minor})${versi.length > 1 ? `  [terpasang juga: ${versi.slice(1).map((x) => x.v.join(".")).join(", ")}]` : ""}`);
if (cap >= 12 && (mayor < 12 || (mayor === 12 && minor < 8))) {
  console.error(`GPU Blackwell (compute ${cap}) butuh CUDA ≥ 12.8, yang terbaru terpasang ${mayor}.${minor}.
Pasang:  sudo apt-get install -y cuda-toolkit-12-9   (atau jalankan dari laptop: bun run server -- --siapkan)`);
  process.exit(1);
}
const arch = Number.isFinite(cap) ? `-arch=sm_${Math.round(cap * 10)}` : "-arch=native";
mkdirSync("build", { recursive: true });
const cmd = [nvcc, "-O3", arch, "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-lineinfo", SUMBER, "-o", FILE_LIB];
console.log(`$ ${cmd.join(" ")}`);
const t0 = performance.now();
const kode = await Bun.spawn(cmd, { stdout: "inherit", stderr: "inherit" }).exited;
if (kode !== 0) { console.error("Kompilasi gagal."); process.exit(1); }
console.log(`Selesai dalam ${((performance.now() - t0) / 1000).toFixed(1)} detik: ${FILE_LIB} (${(statSync(FILE_LIB).size / 1e6).toFixed(1)} MB)`);
console.log("Lanjut: bun run test-gpu");

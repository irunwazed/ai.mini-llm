/**
 * TRAINING DI SERVER (mis. pod packet.ai) lewat SSH — kode tetap sama, hanya dijalankan di mesin lain.
 * Training memakai GPU NVIDIA jika server punya (backend CUDA tulisan sendiri, `train-gpt --gpu`), selain itu CPU.
 *
 * KUNCI SSH (sesuai docs packet.ai: kunci di akun dipasang ke pod yang dibuat SESUDAH kunci ditambahkan):
 *   bun run server -- --buat-kunci [--daftarkan]   buat ~/.ssh/packet_ai (ed25519); --daftarkan = unggah kunci publik
 *                                                  lewat API packet.ai (env PACKET_API_KEY, mis. di .env)
 *   bun run server -- --pasang-kunci               pod yang SUDAH berjalan: pasang kunci pakai password pod sekali
 *   bun run server -- --atur ubuntu@1.2.3.4 --port 30123 [--kunci ~/.ssh/packet_ai]   simpan alamat (Host & Port dari console)
 *   bun run server -- --uji                        cek login SSH + CPU/RAM/GPU
 *   bun run server -- --web [--port 3100]          jalankan web (chat + /latih) DI SERVER, buka di laptop lewat terowongan SSH
 *   bun run server -- --web --hentikan             matikan web di server
 *   bun run server -- --siapkan        pasang bun + tmux, kirim kode, bun install, salin login Drive, cek CPU/RAM
 *   bun run server -- --kirim-kode     kirim ulang kode saja (setelah mengubah kode)
 *   bun run server -- --latih [--jenis reader] [--lewat drive|scp] -- --dataset wiki-v1 --name wiki3 --size besar --minutes 240
 *                                      kirim data -> latih di tmux (tetap jalan walau SSH putus) -> model dikirim balik ke Drive
 *   bun run server -- --status         progres semua proses di server + CPU/RAM
 *   bun run server -- --log wiki3 [--ikuti]
 *   bun run server -- --hentikan wiki3 [--jenis reader]   berhenti dengan aman (checkpoint disimpan)
 *   bun run server -- --ambil wiki3 [--lewat drive|scp]   model dari server ke laptop
 *   bun run server -- --perintah "eval-konteks -- --name wiki3"   jalankan script package.json di server
 *   Tambahkan --kering untuk melihat perintah/skrip tanpa menjalankannya.
 *
 * Data: --lewat drive (bawaan jika sudah login Drive) = laptop unggah ke Drive, server unduh dari Drive;
 *       --lewat scp = langsung disalin lewat SSH.
 * Alamat disimpan di .rahasia/server.json. Memakai ssh/scp/tar bawaan sistem.
 */
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { statusLogin } from "../cloud/google-auth.js";
import { LOKASI } from "../paths.js";
import { argumen } from "./args.js";

// argumen setelah "--" kedua = argumen untuk perintah training
const mentah = process.argv.slice(2);
const pemisah = mentah.indexOf("--");
const argLatih = pemisah >= 0 ? mentah.slice(pemisah + 1) : [];
const { opsi } = argumen(pemisah >= 0 ? mentah.slice(0, pemisah) : mentah);
const kering = opsi.kering === "true";

const FILE_SERVER = join(LOKASI.rahasia, "server.json");
interface Server { tujuan: string; port: number; kunci?: string; folder: string; gpu?: string }
const KUNCI_BAWAAN = join(homedir(), ".ssh", "packet_ai");
const simpanServer = (s: Server) => { mkdirSync(LOKASI.rahasia, { recursive: true }); writeFileSync(FILE_SERVER, JSON.stringify(s, null, 1)); };
const NAMA = /^[\w.-]{1,40}$/;
const ARG_AMAN = /^[\w.,:=\/@+-]*$/;
const sq = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;

function bacaServer(): Server {
  if (!existsSync(FILE_SERVER)) {
    console.error("Alamat server belum diatur. Contoh (lihat alamat SSH di dashboard packet.ai atau `packet ssh`):\n  bun run server -- --atur ubuntu@1.2.3.4 --port 22");
    process.exit(1);
  }
  return JSON.parse(readFileSync(FILE_SERVER, "utf8")) as Server;
}

function argSsh(s: Server, tty = false): string[] {
  return [
    "-p", String(s.port), ...(s.kunci ? ["-i", s.kunci] : []), ...(tty ? ["-t"] : []),
    "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
    ...(process.stdin.isTTY ? [] : ["-o", "BatchMode=yes"]), // dari panel web: jangan menunggu ketikan
  ];
}

/** Jalankan skrip bash di server (dikirim lewat stdin -> tanpa masalah kutip). */
async function jalankanRemote(s: Server, skrip: string, judul: string, tty = false): Promise<number> {
  const lengkap = `set -o pipefail\nexport PATH="$HOME/.bun/bin:/usr/local/cuda/bin:$PATH"\n${skrip}\n`;
  if (kering) { console.log(`--- [kering] ${judul}: ssh ${s.tujuan} bash -s <<\n${lengkap}---`); return 0; }
  console.log(`» ${judul}`);
  const p = tty
    ? Bun.spawn(["ssh", ...argSsh(s, true), s.tujuan, skrip.replace(/\n/g, "; ")], { stdin: "inherit", stdout: "inherit", stderr: "inherit" })
    : Bun.spawn(["ssh", ...argSsh(s), s.tujuan, "bash -s"], { stdin: new TextEncoder().encode(lengkap), stdout: "inherit", stderr: "inherit" });
  return await p.exited;
}

async function jalankanLokal(cmd: string[], judul: string): Promise<number> {
  if (kering) { console.log(`--- [kering] ${judul}: ${cmd.join(" ")}`); return 0; }
  console.log(`» ${judul}`);
  return await Bun.spawn(cmd, { stdin: "inherit", stdout: "inherit", stderr: "inherit" }).exited;
}

async function wajib(kode: number | Promise<number>, apa: string): Promise<void> {
  const k = await kode;
  if (k !== 0) { console.error(`Gagal: ${apa} (kode ${k}).`); process.exit(1); }
}

/** tar daftar path (relatif dari folder proyek) -> salin ke server -> ekstrak di folder proyek server. */
async function kirimPath(s: Server, daftar: string[], judul: string): Promise<void> {
  const ada = daftar.filter((p) => existsSync(p));
  if (!ada.length) { console.log(`  (tidak ada yang dikirim untuk ${judul})`); return; }
  mkdirSync(LOKASI.progres, { recursive: true });
  const fileDaftar = join(LOKASI.progres, "kirim-daftar.txt").replace(/\\/g, "/"), arsip = join(LOKASI.progres, "kirim.tgz").replace(/\\/g, "/");
  writeFileSync(fileDaftar, ada.join("\n") + "\n");
  await wajib(jalankanLokal(["tar", "-czf", arsip, "-T", fileDaftar], `${judul}: membungkus ${ada.length} path`), "membuat arsip");
  await wajib(jalankanLokal(["scp", "-P", String(s.port), ...(s.kunci ? ["-i", s.kunci] : []), "-o", "StrictHostKeyChecking=accept-new", arsip, `${s.tujuan}:kirim.tgz`], `${judul}: menyalin ke server`), "scp");
  await wajib(jalankanRemote(s, `mkdir -p ~/${s.folder} && tar -xzf ~/kirim.tgz -C ~/${s.folder} && rm -f ~/kirim.tgz`, `${judul}: membuka di server`), "ekstrak di server");
  rmSync(arsip, { force: true }); rmSync(fileDaftar, { force: true });
}

/** File kode untuk server: yang dilacak git + file baru yang tidak di-gitignore; tanpa data besar & rahasia. */
async function daftarKode(): Promise<string[]> {
  const p = Bun.spawn(["git", "ls-files", "-co", "--exclude-standard"], { stdout: "pipe" });
  const semua = (await new Response(p.stdout).text()).split("\n").map((x) => x.trim()).filter(Boolean);
  return semua.filter((f) => existsSync(f) && !f.startsWith(".rahasia/") && !f.startsWith("progres/")
    && (!f.startsWith("documents/") || f.startsWith("documents/samples/")) // dokumen datang lewat Drive/scp, kecuali contoh untuk eval
    && !f.startsWith("data/mentah/") && !f.startsWith("data/siap/"));
}

async function kirimKode(s: Server): Promise<void> {
  const f = await daftarKode();
  await kirimPath(s, f, `kode (${f.length} file)`);
  await wajib(jalankanRemote(s, `cd ~/${s.folder} && bun install`, "bun install di server"), "bun install");
}

const pakaiDrive = () => (opsi.lewat ? opsi.lewat === "drive" : statusLogin().login);

// ---------------------------------------------------------------- perintah
/**
 * Isi file .env (dibaca langsung, bukan process.env: di Windows variabel USERNAME milik sistem menimpa .env).
 * Kunci yang dipakai: HOST, PORT, USERNAME, PASSWORD (atau PACKET_SSH_HOST/PORT/USER/PASSWORD).
 */
function bacaDotEnv(): Record<string, string> {
  if (!existsSync(".env")) return {};
  const hasil: Record<string, string> = {};
  for (const baris of readFileSync(".env", "utf8").split(/\r?\n/)) {
    const m = baris.match(/^\s*(?:export\s+)?([A-Za-z_][\w]*)\s*=\s*(.*?)\s*$/);
    if (m) hasil[m[1]] = m[2].replace(/^(['"])(.*)\1$/, "$2");
  }
  return hasil;
}

async function atur(): Promise<void> {
  // --atur tanpa nilai = ambil Host/Port/Username dari .env
  const env = bacaDotEnv();
  if (opsi.atur === "true") {
    const host = env.PACKET_SSH_HOST ?? env.HOST, user = env.PACKET_SSH_USER ?? env.USERNAME ?? "ubuntu";
    if (!host) { console.error("--atur tanpa alamat: isi HOST, PORT, USERNAME di .env, atau tulis --atur ubuntu@<Host> --port <Port>"); process.exit(1); }
    opsi.atur = `${user}@${host}`;
    opsi.port ??= env.PACKET_SSH_PORT ?? env.PORT;
    console.log(`Alamat dari .env: ${opsi.atur} port ${opsi.port ?? 22}`);
  }
  const tujuan = opsi.atur;
  if (!/^[\w.-]+@[\w.-]+$/.test(tujuan)) { console.error("Format: --atur user@host   (mis. ubuntu@203.0.113.5)"); process.exit(1); }
  const port = Number(opsi.port ?? 22);
  if (!Number.isInteger(port) || port < 1 || port > 65535) { console.error("--port tidak valid"); process.exit(1); }
  const kunci = opsi.kunci ?? (existsSync(KUNCI_BAWAAN) ? KUNCI_BAWAAN : undefined);
  const s: Server = { tujuan, port, folder: opsi.folder && NAMA.test(opsi.folder) ? opsi.folder : "llm", ...(kunci ? { kunci } : {}) };
  simpanServer(s);
  // pod baru sering mendapat host:port yang pernah dipakai pod lain -> host key lama harus dibuang (docs: Troubleshooting)
  const host = tujuan.split("@")[1], entri = port === 22 ? host : `[${host}]:${port}`;
  if (!kering && existsSync(join(homedir(), ".ssh", "known_hosts"))) await Bun.spawn(["ssh-keygen", "-R", entri], { stdout: "ignore", stderr: "ignore" }).exited;
  tulisSshConfig(s);
  console.log(`Disimpan: ${s.tujuan} port ${s.port}${s.kunci ? `, kunci ${s.kunci}` : " (tanpa kunci: akan ditanya password)"}, folder ~/${s.folder}`);
  console.log(`Juga bisa: ssh packet-llm   (blok "Host packet-llm" di ~/.ssh/config; dipakai juga oleh VS Code Remote SSH)`);
  console.log(`Lanjut:  bun run server -- --uji   lalu   bun run server -- --siapkan`);
}

/** Blok "Host packet-llm" di ~/.ssh/config, di antara penanda agar bisa diperbarui tanpa menyentuh isi lain. */
function tulisSshConfig(s: Server): void {
  if (kering) return;
  const f = join(homedir(), ".ssh", "config");
  const awal = "# >>> llm-dari-nol (bun run server -- --atur)", akhir = "# <<< llm-dari-nol";
  const [user, host] = s.tujuan.split("@");
  const blok = [awal, "Host packet-llm", `    HostName ${host}`, `    Port ${s.port}`, `    User ${user}`, ...(s.kunci ? [`    IdentityFile ${s.kunci.replace(/\\/g, "/")}`] : []), "    ServerAliveInterval 30", akhir].join("\n");
  mkdirSync(join(homedir(), ".ssh"), { recursive: true });
  const lama = existsSync(f) ? readFileSync(f, "utf8") : "";
  const i = lama.indexOf(awal), j = lama.indexOf(akhir);
  const baru = i >= 0 && j > i ? lama.slice(0, i) + blok + lama.slice(j + akhir.length) : (lama.trimEnd() ? lama.trimEnd() + "\n\n" : "") + blok + "\n";
  writeFileSync(f, baru);
}

// ---------------------------------------------------------------- kunci SSH
async function buatKunci(): Promise<void> {
  const pub = KUNCI_BAWAAN + ".pub";
  if (existsSync(KUNCI_BAWAAN)) console.log(`Kunci sudah ada: ${KUNCI_BAWAAN} (tidak ditimpa)`);
  else {
    mkdirSync(join(homedir(), ".ssh"), { recursive: true });
    console.log("Membuat kunci SSH ed25519 (boleh diberi passphrase; kosongkan = tanpa passphrase)…");
    await wajib(jalankanLokal(["ssh-keygen", "-t", "ed25519", "-f", KUNCI_BAWAAN, "-C", "packet.ai llm-dari-nol", ...(process.stdin.isTTY ? [] : ["-N", ""])], "ssh-keygen"), "membuat kunci");
  }
  if (kering) return;
  if (process.platform !== "win32") { await Bun.spawn(["chmod", "600", KUNCI_BAWAAN]).exited; await Bun.spawn(["chmod", "644", pub]).exited; }
  const isi = readFileSync(pub, "utf8").trim();
  console.log(`\nKunci PUBLIK (aman dibagikan; kunci privat ${KUNCI_BAWAAN} JANGAN pernah dibagikan):\n\n${isi}\n`);
  if (opsi.daftarkan === "true") { await daftarkanKunci(isi); return; }
  console.log(`Pasang di packet.ai (pilih salah satu):
  a) Console -> Account -> API keys -> bagian "SSH keys" -> Add key -> tempel baris di atas.
  b) Otomatis lewat API:  isi PACKET_API_KEY=pk_live_... di file .env, lalu  bun run server -- --buat-kunci --daftarkan
Kunci hanya terpasang di pod yang dibuat SESUDAH kunci ditambahkan. Pod yang sudah berjalan:  bun run server -- --pasang-kunci
Lalu:  bun run server -- --atur ubuntu@<Host> --port <Port>   (Host & Port dari Compute -> Running -> Terminal)`);
}

async function daftarkanKunci(publik: string): Promise<void> {
  const kunciApi = process.env.PACKET_API_KEY ?? "";
  if (!/^pk_[\w-]+$/.test(kunciApi)) { console.error("PACKET_API_KEY belum diisi (format pk_live_…). Tambahkan ke file .env (sudah di-gitignore) lalu ulangi."); process.exit(1); }
  const API = "https://dash.packet.ai/api/v2/account/ssh-keys";
  const kepala = { authorization: `Bearer ${kunciApi}`, "content-type": "application/json" };
  // sidik kunci lokal untuk mencegah dobel
  const sidik = (await new Response(Bun.spawn(["ssh-keygen", "-lf", KUNCI_BAWAAN + ".pub"], { stdout: "pipe" }).stdout).text()).split(" ")[1] ?? "";
  const daftar = await fetch(API, { headers: kepala }).then(async (r) => (r.ok ? r.json() : Promise.reject(new Error(`daftar kunci gagal: HTTP ${r.status}`)))).catch((e: Error) => { console.error(e.message); process.exit(1); });
  const kunci = (Array.isArray(daftar) ? daftar : (daftar as { keys?: unknown[]; data?: unknown[]; sshKeys?: unknown[] }).keys ?? (daftar as { data?: unknown[] }).data ?? (daftar as { sshKeys?: unknown[] }).sshKeys ?? []) as { name?: string; fingerprint?: string }[];
  if (sidik && kunci.some((k) => k.fingerprint && (k.fingerprint === sidik || k.fingerprint.endsWith(sidik.replace(/^SHA256:/, ""))))) { console.log("Kunci ini sudah terdaftar di akun packet.ai."); return; }
  const r = await fetch(API, { method: "POST", headers: kepala, body: JSON.stringify({ name: "llm-dari-nol", publicKey: publik }) });
  if (r.ok) console.log("Kunci publik terdaftar di akun packet.ai. Pod yang Anda buat setelah ini otomatis memakai kunci ini.");
  else if (r.status === 409) console.log("Kunci ini sudah terdaftar (sidik sama).");
  else { console.error(`Gagal mendaftarkan kunci: HTTP ${r.status} ${(await r.text()).slice(0, 200)}`); process.exit(1); }
}

/** Pod yang sudah berjalan: pasang kunci publik memakai password pod (ditanya sekali). */
async function pasangKunci(s: Server): Promise<void> {
  const pub = (s.kunci ?? KUNCI_BAWAAN) + ".pub";
  if (!existsSync(pub)) { console.error(`Kunci publik ${pub} belum ada. Buat dulu: bun run server -- --buat-kunci`); process.exit(1); }
  // password dari .env (PASSWORD / PACKET_SSH_PASSWORD) dipakai lewat SSH_ASKPASS: tidak dicetak, tidak ditulis ke disk
  const sandi = bacaDotEnv().PACKET_SSH_PASSWORD ?? bacaDotEnv().PASSWORD;
  const envSsh: Record<string, string | undefined> = { ...process.env };
  let askpass: string | null = null;
  if (sandi) {
    askpass = join(tmpdir(), `llm-askpass-${process.pid}.sh`);
    writeFileSync(askpass, `#!/bin/sh\nprintf '%s\\n' "$LLM_SSH_SANDI"\n`, { mode: 0o700 });
    Object.assign(envSsh, { SSH_ASKPASS: askpass, SSH_ASKPASS_REQUIRE: "force", DISPLAY: envSsh.DISPLAY ?? "llm:0", LLM_SSH_SANDI: sandi });
    console.log("Memakai password pod dari .env (tidak dicetak).");
  } else console.log("Masukkan PASSWORD pod saat diminta (salin dari Compute -> Running -> Terminal -> SSH connection details).");
  // kunci ditambahkan ke authorized_keys hanya jika belum ada (aman dijalankan ulang)
  const remote = `mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && k="$(cat)" && { grep -qxF "$k" ~/.ssh/authorized_keys || printf '%s\\n' "$k" >> ~/.ssh/authorized_keys; } && chmod 600 ~/.ssh/authorized_keys && echo "Kunci publik terpasang di $(whoami)@$(hostname)."`;
  let kode = 0;
  if (kering) console.log(`--- [kering] ssh -p ${s.port} ${s.tujuan} <${pub}  ${remote}`);
  else {
    try {
      kode = await Bun.spawn(["ssh", "-p", String(s.port), "-o", "StrictHostKeyChecking=accept-new", "-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password,keyboard-interactive", "-o", "ConnectTimeout=20", s.tujuan, remote],
        { stdin: Bun.file(pub), stdout: "inherit", stderr: "inherit", env: envSsh }).exited;
    } finally { if (askpass) rmSync(askpass, { force: true }); }
  }
  if (kode !== 0) { console.error("Gagal memasang kunci. Pastikan password benar dan pod berstatus running."); process.exit(1); }
  if (!s.kunci && existsSync(s.kunci ?? KUNCI_BAWAAN)) { s.kunci = KUNCI_BAWAAN; simpanServer(s); tulisSshConfig(s); }
  console.log("Kunci terpasang. Selanjutnya tanpa password:  bun run server -- --uji");
}

/** Cek login SSH tanpa password + spesifikasi; galat diterjemahkan sesuai docs packet.ai (Troubleshooting). */
async function uji(s: Server): Promise<void> {
  const skrip = `echo "Masuk sebagai $(whoami)@$(hostname)"
echo "CPU   : $(nproc) core"
free -g | awk '/Mem:/ {print "RAM   : " $2 " GB"}'
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/GPU   : /'; else echo "GPU   : tidak ada"; fi
if command -v nvcc >/dev/null; then echo "CUDA  : $(nvcc --version | grep -o 'release [0-9.]*')"; else echo "CUDA  : nvcc belum ada (dipasang otomatis oleh --siapkan jika perlu)"; fi`;
  if (kering) { console.log(`--- [kering] ssh -o BatchMode=yes ${argSsh(s).join(" ")} ${s.tujuan} bash -s <<\n${skrip}`); return; }
  const p = Bun.spawn(["ssh", ...argSsh(s), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", s.tujuan, "bash -s"], { stdin: new TextEncoder().encode(`export PATH="/usr/local/cuda/bin:$PATH"\n${skrip}\n`), stdout: "inherit", stderr: "pipe" });
  const err = await new Response(p.stderr).text(), kode = await p.exited;
  if (kode === 0) { console.log("\nSSH OK. Lanjut:  bun run server -- --siapkan"); return; }
  process.stderr.write(err);
  if (/Permission denied/i.test(err)) console.error(`\nKunci belum diterima pod ini. Kunci di akun hanya terpasang di pod yang dibuat SESUDAH kunci ditambahkan.
  Pod yang sudah berjalan:  bun run server -- --pasang-kunci   (pakai password pod sekali)`);
  else if (/refused|timed out|No route/i.test(err)) console.error("\nPod belum running, atau Host/Port berubah (setiap pod punya host:port sendiri). Salin ulang dari Compute -> Running -> Terminal, lalu --atur lagi.");
  else if (/REMOTE HOST IDENTIFICATION HAS CHANGED|Host key verification failed/i.test(err)) console.error("\nHost key berubah (pod baru di host:port lama). Jalankan --atur lagi (host key lama otomatis dibuang).");
  process.exit(1);
}

// ---------------------------------------------------------------- web di server (terowongan SSH)
async function web(s: Server): Promise<void> {
  const port = Number(opsi.port ?? 3100);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) { console.error("--port harus 1024–65535"); process.exit(1); }
  if (opsi.hentikan === "true") { await jalankanRemote(s, `tmux kill-session -t web 2>/dev/null && echo "Web di server dimatikan." || echo "Web tidak sedang berjalan."`, "mematikan web di server"); return; }
  const skrip = `cd ~/${s.folder} 2>/dev/null || { echo "Belum disiapkan: bun run server -- --siapkan"; exit 1; }
mkdir -p progres/log
if [ ! -f model/indeks.json ]; then echo "model/ belum ada: melatih indeks chatbot aturan dulu…"; bun run train || exit 1; fi
if tmux has-session -t web 2>/dev/null; then echo "Web sudah berjalan di server."
else
cat > progres/web.sh <<'AKHIR_WEB'
export PATH="$HOME/.bun/bin:/usr/local/cuda/bin:$PATH"
cd ~/${s.folder}
bun run web -- --port ${port} 2>&1 | tee -a progres/log/server-web.log
AKHIR_WEB
tmux new-session -d -s web "bash ~/${s.folder}/progres/web.sh"
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:${port}/latih && break; sleep 1; done
fi
curl -s -o /dev/null -w "Web di server menjawab (HTTP %{http_code}).\\n" http://127.0.0.1:${port}/latih || { echo "Web belum menjawab. Lihat: bun run server -- --perintah status / log progres/log/server-web.log"; exit 1; }`;
  await wajib(jalankanRemote(s, skrip, `menjalankan web di server (port ${port}, hanya 127.0.0.1)`), "web di server");
  const url = `http://localhost:${port}/latih`;
  console.log(`\nTerowongan SSH: laptop :${port} -> server 127.0.0.1:${port}. Buka  ${url}
(Biarkan jendela ini terbuka. Ctrl+C = tutup terowongan; web di server tetap jalan. Matikan: bun run server -- --web --hentikan)`);
  if (kering) { console.log(`--- [kering] ssh -N -L ${port}:localhost:${port} ${argSsh(s).join(" ")} ${s.tujuan}`); return; }
  setTimeout(() => {
    if (process.platform === "win32") Bun.spawn(["rundll32", "url.dll,FileProtocolHandler", url], { stdout: "ignore", stderr: "ignore" });
    else if (process.platform === "darwin") Bun.spawn(["open", url], { stdout: "ignore", stderr: "ignore" });
  }, 2500);
  const kode = await Bun.spawn(["ssh", "-N", "-L", `${port}:localhost:${port}`, "-o", "ExitOnForwardFailure=yes", ...argSsh(s), s.tujuan], { stdin: "inherit", stdout: "inherit", stderr: "inherit" }).exited;
  if (kode !== 0) console.error(`Terowongan gagal (kode ${kode}). Port ${port} di laptop mungkin sudah dipakai: coba --port lain.`);
}

async function siapkan(s: Server): Promise<void> {
  await wajib(jalankanRemote(s, `set -e
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=sudo
if ! command -v tmux >/dev/null || ! command -v unzip >/dev/null || ! command -v curl >/dev/null; then
  $SUDO apt-get update -y -qq && $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux unzip curl
fi
if [ ! -x "$HOME/.bun/bin/bun" ]; then curl -fsSL https://bun.sh/install | bash; fi
"$HOME/.bun/bin/bun" --version`, "memasang bun + tmux"), "pemasangan");
  await kirimKode(s);
  const rahasia = ["google-oauth.json", "google-token.json"].map((f) => join(LOKASI.rahasia, f).replace(/\\/g, "/")).filter((f) => existsSync(f));
  if (rahasia.length && opsi["tanpa-drive"] !== "true") {
    await kirimPath(s, rahasia, "login Google Drive (token, hanya scope drive.file)");
    await jalankanRemote(s, `chmod 700 ~/${s.folder}/.rahasia && chmod 600 ~/${s.folder}/.rahasia/*`, "mengunci izin file token");
  } else if (!rahasia.length) console.log("  (belum login Drive di laptop: data akan dikirim lewat scp. Login dulu jika ingin lewat Drive: bun run drive -- --login)");
  await jalankanRemote(s, `echo "CPU   : $(nproc) core"
free -g | awk '/Mem:/ {print "RAM   : " $2 " GB total, " $7 " GB tersedia"}'
echo "Disk  : $(df -h ~ | awk 'NR==2 {print $4 " kosong"}')"`, "spesifikasi server");
  // GPU: deteksi -> pasang nvcc jika perlu -> kompilasi backend CUDA (bun run build-gpu)
  const gpu = kering ? "" : (await new Response(Bun.spawn(["ssh", ...argSsh(s), "-o", "BatchMode=yes", s.tujuan, "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1"], { stdout: "pipe", stderr: "ignore" }).stdout).text()).trim();
  if (gpu || kering) {
    console.log(`GPU   : ${gpu || "(kering)"} -> menyiapkan backend CUDA`);
    const kode = await jalankanRemote(s, `cd ~/${s.folder}
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=sudo
# versi nvcc tertinggi yang terpasang (0 jika tidak ada)
versi_nvcc() { best=0; for n in $(command -v nvcc) /usr/local/cuda*/bin/nvcc; do [ -x "$n" ] || continue; v=$("$n" --version | grep -o 'release [0-9]*\\.[0-9]*' | awk '{split($2,a,"."); print a[1]*100+a[2]}'); [ "\${v:-0}" -gt "$best" ] && best=$v; done; echo $best; }
# GPU Blackwell (RTX 50xx dll.) butuh CUDA >= 12.8
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiE 'RTX 50[0-9]{2}|Blackwell|B[12]00'; then perlu=1208; else perlu=1100; fi
if [ "$(versi_nvcc)" -lt "$perlu" ]; then
  echo "CUDA toolkit terpasang terlalu lama/tidak ada untuk GPU ini: memasang cuda-toolkit-12-9 (sekali, beberapa menit)…"
  # repo NVIDIA sudah ada (banyak pod GPU sudah memilikinya)? jangan tambah keyring kedua (bentrok Signed-By)
  if ! grep -rqs "developer.download.nvidia.com/compute/cuda" /etc/apt/sources.list /etc/apt/sources.list.d/; then
    curl -fsSLo /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && $SUDO dpkg -i /tmp/cuda-keyring.deb
  fi
  $SUDO apt-get update -y -qq && $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-12-9 || { echo "Gagal memasang CUDA 12.9"; exit 1; }
fi
bun run build-gpu`, "CUDA toolkit + kompilasi backend CUDA (build-gpu)");
    if (kode === 0 && !kering) { s.gpu = gpu; simpanServer(s); console.log(`\nServer siap dengan GPU ${gpu}. --latih otomatis memakai --gpu.`); }
    else if (!kering) console.log("\nBackend CUDA gagal dikompilasi (lihat pesan di atas). Training tetap bisa memakai CPU server.");
  } else console.log("\nServer tanpa GPU NVIDIA: training memakai CPU (kecepatan = jumlah core).");
  console.log(`Contoh:  bun run server -- --latih -- --dataset obrolan-v1 --from wiki2 --name chat3 --steps 1500
UI di server:  bun run server -- --web`);
}

function nilaiArg(nama: string): string | undefined { const i = argLatih.indexOf(`--${nama}`); return i >= 0 ? argLatih[i + 1] : undefined; }

async function latih(s: Server): Promise<void> {
  const jenis = opsi.jenis === "reader" ? "train-reader" : "train-gpt";
  const salah = argLatih.filter((a) => !ARG_AMAN.test(a));
  if (salah.length) { console.error(`Argumen tidak aman/didukung: ${salah.join(" ")}`); process.exit(1); }
  const nama = nilaiArg("name"), dataset = nilaiArg("dataset"), dari = nilaiArg("from"), resume = argLatih.includes("--resume");
  if (!nama || !NAMA.test(nama)) { console.error("Wajib: -- --name <model>   (contoh: bun run server -- --latih -- --dataset wiki-v1 --name wiki3 --size besar --minutes 240)"); process.exit(1); }
  for (const x of [dataset, dari]) if (x !== undefined && !NAMA.test(x)) { console.error(`Nama tidak valid: ${x}`); process.exit(1); }
  if (jenis === "train-gpt" && !dataset && !nilaiArg("data")) { console.error("Untuk server, pakai dataset siap latih: -- --dataset <nama>  (siapkan dulu: bun run siapkan-data)"); process.exit(1); }

  // data yang dibutuhkan
  const kebutuhan = [
    ...(dataset ? [`${LOKASI.siap}/${dataset}`] : []),
    ...(dari ? [`${LOKASI.modelGPT}/${dari}`] : []),
    ...(resume ? [`${LOKASI.modelGPT}/${nama}`] : []),
    ...(jenis === "train-reader" ? [LOKASI.pembaca.replace(/\/[^/]+$/, "")] : []),
  ];
  const kurang = kebutuhan.filter((p) => !existsSync(p));
  if (kurang.length) { console.error(`Tidak ada di laptop: ${kurang.join(", ")}`); process.exit(1); }
  const drive = pakaiDrive();
  console.log(`Data dikirim lewat ${drive ? "Google Drive" : "scp"}: ${kebutuhan.join(", ") || "(tidak ada)"}`);
  if (kebutuhan.length) {
    if (drive) await wajib(jalankanLokal(["bun", "src/commands/drive.ts", "--unggah", "--folder", kebutuhan.join(",")], "unggah data ke Drive"), "unggah ke Drive");
    else await kirimPath(s, kebutuhan, "data");
  }

  if (jenis === "train-gpt" && s.gpu && !argLatih.includes("--gpu") && opsi.cpu !== "true") { argLatih.push("--gpu"); console.log(`GPU server (${s.gpu}) dipakai: --gpu  (paksa CPU dengan --cpu)`); }
  const sesi = `latih-${nama}`, log = `progres/log/server-${nama}.log`;
  const unduh = drive && kebutuhan.length ? `bun run drive -- --unduh --folder ${sq(kebutuhan.join(","))} || { echo "unduh dari Drive gagal"; exit 1; }` : "";
  const unggahBalik = drive ? `bun run drive -- --unggah --folder ${sq(`${LOKASI.modelGPT}/${nama}`)} || echo "PERINGATAN: model belum terunggah ke Drive (ambil dengan --lewat scp)"` : "";
  const skrip = `#!/bin/bash
export PATH="$HOME/.bun/bin:/usr/local/cuda/bin:$PATH"
cd ~/${s.folder}
mkdir -p progres/log
exec > >(tee -a ${log}) 2>&1
echo "== mulai $(date '+%F %T') · $(nproc) core"
${unduh}
bun run ${jenis} -- ${argLatih.map(sq).join(" ")}
kode=$?
${unggahBalik}
echo "== selesai $(date '+%F %T') · kode keluar $kode"
`;
  await wajib(jalankanRemote(s, `cd ~/${s.folder} && mkdir -p progres/log
if tmux has-session -t ${sq(sesi)} 2>/dev/null; then echo "Training ${nama} masih berjalan di server (tmux ${sesi}). Hentikan dulu: bun run server -- --hentikan ${nama}"; exit 3; fi
cat > progres/${sesi}.sh <<'AKHIR_SKRIP_LATIH'
${skrip}AKHIR_SKRIP_LATIH
tmux new-session -d -s ${sq(sesi)} "bash ~/${s.folder}/progres/${sesi}.sh"
echo "Dimulai di tmux sesi ${sesi}."`, `memulai ${jenis} ${nama} di server`), "memulai training");
  console.log(`
Training berjalan di server (tetap jalan walau laptop/SSH putus).
  pantau   : bun run server -- --status        atau  bun run server -- --log ${nama} --ikuti
  hentikan : bun run server -- --hentikan ${nama}${jenis === "train-reader" ? " --jenis reader" : ""}   (aman, checkpoint disimpan)
  ambil    : bun run server -- --ambil ${nama}${drive ? "   (model juga otomatis diunggah ke Drive saat selesai)" : ""}`);
}

async function ambil(s: Server): Promise<void> {
  const nama = opsi.ambil;
  if (!NAMA.test(nama)) { console.error("Format: --ambil <nama-model>"); process.exit(1); }
  const folder = `${LOKASI.modelGPT}/${nama}`;
  if (pakaiDrive()) {
    await wajib(jalankanLokal(["bun", "src/commands/drive.ts", "--unduh", "--folder", folder], `unduh ${folder} dari Drive`), "unduh dari Drive");
  } else {
    mkdirSync(LOKASI.progres, { recursive: true });
    const arsip = `${LOKASI.progres}/ambil-${nama}.tgz`;
    await wajib(jalankanRemote(s, `cd ~/${s.folder} && test -d ${sq(folder)} && tar -czf ~/ambil.tgz ${sq(folder)}`, `membungkus ${folder} di server`), "model tidak ada di server");
    await wajib(jalankanLokal(["scp", "-P", String(s.port), ...(s.kunci ? ["-i", s.kunci] : []), `${s.tujuan}:ambil.tgz`, arsip], "menyalin ke laptop"), "scp");
    await wajib(jalankanLokal(["tar", "-xzf", arsip], "membuka arsip"), "ekstrak");
    if (!kering) rmSync(arsip, { force: true });
  }
  console.log(`Model ada di ${folder}/. Pilih di halaman chat (GPT obrolan/pembaca) atau: bun run eval-konteks -- --name ${nama}`);
}

async function perintah(s: Server): Promise<void> {
  const bagian = opsi.perintah.trim().split(/\s+/);
  const scripts = (JSON.parse(readFileSync("package.json", "utf8")) as { scripts: Record<string, string> }).scripts;
  if (!scripts[bagian[0]]) { console.error(`Script tidak dikenal: ${bagian[0]}`); process.exit(1); }
  const salah = bagian.slice(1).filter((a) => !ARG_AMAN.test(a));
  if (salah.length) { console.error(`Argumen tidak aman/didukung: ${salah.join(" ")}`); process.exit(1); }
  await jalankanRemote(s, `cd ~/${s.folder} && bun run ${bagian.map(sq).join(" ")}`, `bun run ${bagian.join(" ")} di server`);
}

// ---------------------------------------------------------------- utama
if (opsi.atur) await atur();
else if (opsi["buat-kunci"] === "true") await buatKunci();
else {
  const s = bacaServer();
  if (opsi.uji === "true") await uji(s);
  else if (opsi["pasang-kunci"] === "true") await pasangKunci(s);
  else if (opsi.web === "true") await web(s);
  else if (opsi.siapkan === "true") await siapkan(s);
  else if (opsi["kirim-kode"] === "true") await kirimKode(s);
  else if (opsi.latih) await latih(s);
  else if (opsi.ambil) await ambil(s);
  else if (opsi.perintah) await perintah(s);
  else if (opsi.hentikan) {
    if (!NAMA.test(opsi.hentikan)) { console.error("Format: --hentikan <nama-model>"); process.exit(1); }
    const jenis = opsi.jenis === "reader" ? "train-reader" : "train-gpt";
    await jalankanRemote(s, `cd ~/${s.folder} && bun run status -- --hentikan ${sq(`${jenis}:${opsi.hentikan}`)}`, `menghentikan ${opsi.hentikan} dengan aman`);
  } else if (opsi.log) {
    if (!NAMA.test(opsi.log)) { console.error("Format: --log <nama-model>"); process.exit(1); }
    const f = `~/${s.folder}/progres/log/server-${opsi.log}.log`;
    await jalankanRemote(s, opsi.ikuti === "true" ? `tail -n 40 -f ${f}` : `tail -n 80 ${f}`, `log ${opsi.log}`, opsi.ikuti === "true");
  } else {
    await jalankanRemote(s, `cd ~/${s.folder} 2>/dev/null || { echo "Belum disiapkan: bun run server -- --siapkan"; exit 1; }
bun run status
echo; echo "Sesi tmux:"; tmux ls 2>/dev/null || echo "  (tidak ada)"
echo; echo "Beban : $(uptime | sed 's/.*load average/load average/') · $(nproc) core"
free -g | awk '/Mem:/ {print "RAM   : " $3 " / " $2 " GB terpakai"}'`, `status ${s.tujuan}`);
  }
}

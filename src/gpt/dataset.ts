/**
 * DATA LATIH GPT.
 *
 * Sumber (boleh beberapa, dipisah koma: "data/wiki-id.txt,documents"):
 *   - folder dokumen (PDF/DOCX/TXT/MD) -> dibaca dengan pembaca yang sama seperti chatbot,
 *     header/footer dibuang dan kesalahan OCR umum diperbaiki
 *   - file teks (.txt) -> teks bebas untuk PRETRAINING
 *   - file .jsonl -> PERCAKAPAN untuk fine-tuning chat, satu percakapan per baris:
 *       {"turns":[{"user":"halo"},{"bot":"Halo! Ada yang bisa saya bantu?"}]}
 *
 * Mode teks: pasangan latih = jendela acak sepanjang T dari aliran token:
 *   x = token[s .. s+T−1],  y = token[s+1 .. s+T]   (di setiap posisi: tebak token berikutnya)
 * Validasi = setiap blok ke-10 (256 token), BUKAN ekor teks, agar val mewakili seluruh topik.
 *
 * Mode chat: setiap percakapan menjadi satu urutan
 *   <|user|> halo<|end|><|bot|> Halo! Ada yang bisa saya bantu?<|end|>
 * dan target hanya dihitung pada token jawaban bot (token lain diberi −1 = diabaikan).
 */
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { bacaFile, daftarFile, terlihatHasilScan } from "../readers/index.js";
import { bersihkanHalaman, perbaikiOCR } from "../text/clean.js";
import { BOT, END, KONTEKS, USER, type Tokenizer } from "./bpe.js";

export type Giliran = { user?: string; bot?: string; konteks?: string };
export interface Sumber { teks: string; percakapan: Giliran[][]; keterangan: string[] }
export interface Urutan { x: Int32Array; y: Int32Array }

function rapikan(teks: string, hurufKecil: boolean): string {
  const t = hurufKecil ? teks.toLowerCase() : teks;
  return t
    .split(/\r?\n/)
    .map((b) => perbaikiOCR(b.replace(/\s+/g, " ").trim()))
    .filter(Boolean)
    .join("\n");
}

async function bacaSatu(path: string, folderCache: string, hurufKecil: boolean, s: Sumber): Promise<void> {
  if (!existsSync(path)) throw new Error(`Data "${path}" tidak ditemukan.`);
  if (statSync(path).isFile()) {
    if (path.endsWith(".jsonl")) {
      const daftar = readFileSync(path, "utf8").split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l).turns as Giliran[]);
      for (const p of daftar) for (const g of p) {
        if (g.user !== undefined && hurufKecil) g.user = g.user.toLowerCase();
        if (g.bot !== undefined && hurufKecil) g.bot = g.bot.toLowerCase();
      }
      s.percakapan.push(...daftar);
      s.keterangan.push(`${path}: ${daftar.length} percakapan`);
    } else {
      const t = rapikan(readFileSync(path, "utf8"), hurufKecil);
      s.teks += (s.teks ? "\n" : "") + t;
      s.keterangan.push(`${path}: ${t.length.toLocaleString("id-ID")} karakter`);
    }
    return;
  }
  const files = daftarFile(path);
  let dilewati = 0, jumlah = 0;
  for (const f of files) {
    try {
      const { dok } = await bacaFile(f, path, join(folderCache, "cache"));
      if (terlihatHasilScan(dok)) { dilewati++; continue; }
      const halaman = dok.adaHalaman ? bersihkanHalaman(dok.halaman) : dok.halaman;
      const t = rapikan(halaman.join("\n"), hurufKecil);
      s.teks += (s.teks ? "\n" : "") + t;
      jumlah += t.length;
    } catch {
      dilewati++;
    }
  }
  s.keterangan.push(`${path}: ${files.length - dilewati} dokumen, ${jumlah.toLocaleString("id-ID")} karakter${dilewati ? `, ${dilewati} dilewati` : ""}`);
}

export async function bacaSumber(spesifikasi: string, folderCache: string, hurufKecil = true): Promise<Sumber> {
  const s: Sumber = { teks: "", percakapan: [], keterangan: [] };
  for (const p of spesifikasi.split(",").map((x) => x.trim()).filter(Boolean)) await bacaSatu(p, folderCache, hurufKecil, s);
  if (s.teks && s.percakapan.length) throw new Error("Jangan campur teks biasa dan percakapan .jsonl dalam satu training. Pretrain dulu, lalu fine-tune dengan --from.");
  return s;
}

/** Mode teks: aliran token, dibagi train/val per blok 256 token (blok ke-10 = val). */
export function dataTeks(tok: Tokenizer, teks: string): { train: Int32Array; val: Int32Array } {
  const semua = tok.encode(teks);
  const B = 256, train: number[] = [], val: number[] = [];
  for (let i = 0, blok = 0; i < semua.length; i += B, blok++) {
    const potong = semua.slice(i, i + B);
    (blok % 10 === 9 ? val : train).push(...potong);
  }
  return { train: Int32Array.from(train), val: Int32Array.from(val) };
}

/**
 * Satu percakapan dalam bentuk token, BELUM dipotong ke panjang konteks model.
 * potongan[i] = satu giliran: ids token + latih (1 = token jawaban bot yang dilatih).
 * Bentuk inilah yang disimpan oleh `bun run siapkan-data` (data/siap/<dataset>/), supaya dataset yang
 * sama bisa dipakai model dengan konteks berapa pun.
 */
export interface PercakapanToken { kartu: boolean; potongan: { ids: number[]; latih: number[] }[] }

export function encodePercakapan(tok: Tokenizer, giliran: Giliran[]): PercakapanToken {
  const idUser = tok.idToken(USER), idBot = tok.idToken(BOT), idEnd = tok.idToken(END);
  if (idUser === undefined || idBot === undefined || idEnd === undefined)
    throw new Error("Tokenizer model ini belum punya token percakapan. Pretrain ulang dengan versi kode terbaru.");
  const idKonteks = tok.idToken(KONTEKS);
  const potongan = giliran.map((g) => {
    const ids: number[] = [], latih: number[] = [];
    if (g.konteks !== undefined && idKonteks !== undefined) {
      // kartu konteks: <|konteks|> nama: Sari; tinggal: Medan<|end|>  (dibaca, tidak dilatih)
      const k = [idKonteks, ...tok.encode(" " + g.konteks.trim()), idEnd];
      ids.push(...k); latih.push(...k.map(() => 0));
    }
    if (g.user !== undefined) {
      const u = [idUser, ...tok.encode(" " + g.user.trim()), idEnd];
      ids.push(...u); latih.push(...u.map(() => 0));
    }
    if (g.bot !== undefined) {
      ids.push(idBot); latih.push(0);
      const b = [...tok.encode(" " + g.bot.trim()), idEnd];
      ids.push(...b); latih.push(...b.map(() => 1));
    }
    return { ids, latih };
  }).filter((p) => p.ids.length);
  return { kartu: giliran[0]?.konteks !== undefined && idKonteks !== undefined, potongan };
}

/**
 * Potong percakapan agar muat di konteks C: giliran paling awal dibuang, KECUALI kartu konteks
 * (selalu dipertahankan di depan). Target hanya pada token jawaban bot (lainnya −1 = diabaikan).
 */
export function potongKeKonteks(p: PercakapanToken, C: number): Urutan | null {
  const kartu = p.kartu ? p.potongan[0] : null;
  const isi = kartu ? p.potongan.slice(1) : p.potongan;
  let mulai = 0;
  const panjang = () => (kartu?.ids.length ?? 0) + isi.slice(mulai).reduce((s, q) => s + q.ids.length, 0);
  while (mulai < isi.length - 1 && panjang() > C + 1) mulai++;
  const dipakai = kartu ? [kartu, ...isi.slice(mulai)] : isi.slice(mulai);
  const ids = dipakai.flatMap((q) => q.ids).slice(0, C + 1);
  const latih = dipakai.flatMap((q) => q.latih).slice(0, C + 1);
  if (ids.length < 2) return null;
  return { x: Int32Array.from(ids.slice(0, -1)), y: Int32Array.from(ids.slice(1).map((id, i) => (latih[i + 1] ? id : -1))) };
}

/** Bagi percakapan bertoken menjadi latih/val (setiap percakapan ke-10 = val) lalu potong ke konteks C. */
export function urutanChat(daftar: PercakapanToken[], C: number): { train: Urutan[]; val: Urutan[] } {
  const train: Urutan[] = [], val: Urutan[] = [];
  daftar.forEach((p, n) => { const u = potongKeKonteks(p, C); if (u) (n % 10 === 9 ? val : train).push(u); });
  return { train, val };
}

/** Mode chat langsung dari teks percakapan (encode + potong). */
export function dataChat(tok: Tokenizer, percakapan: Giliran[][], C: number): { train: Urutan[]; val: Urutan[] } {
  return urutanChat(percakapan.map((g) => encodePercakapan(tok, g)), C);
}

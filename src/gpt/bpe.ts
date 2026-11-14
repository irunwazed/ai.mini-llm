/**
 * BPE (Byte Pair Encoding) dari nol: cara GPT memecah teks menjadi token.
 *
 * Masalah: huruf saja -> urutan terlalu panjang; kata utuh -> kosakata tak terbatas.
 * BPE mengambil jalan tengah, mulai dari huruf lalu berulang kali menggabungkan
 * PASANGAN simbol yang paling sering muncul:
 *
 *   korpus: " cuti" ×50, " cutinya" ×10
 *   awal  : [" ","c","u","t","i"] ...
 *   merge 1: ("t","i") paling sering   -> "ti"
 *   merge 2: ("u","ti")                -> "uti"
 *   merge 3: (" c","uti") ...           -> " cuti"   (satu token!)
 *
 * Hasil training = daftar merge berurutan. Untuk meng-encode kata baru, merge
 * diterapkan sesuai urutan (peringkat) yang sama.
 *
 * Pra-tokenisasi: teks dipecah dulu menjadi "kata" beserta spasi di depannya
 * (" cuti", " pegawai", ".", "\n"), sehingga merge tidak pernah melintasi batas kata.
 */
export interface BPE {
  vocab: string[];             // id -> string token
  merges: [string, string][];  // urutan merge hasil training
}

export const UNK = "<unk>";
/**
 * Token khusus untuk percakapan. Selalu ada di kosakata dan tidak pernah dipecah:
 *   <|user|> halo<|end|><|bot|> Halo! Ada yang bisa saya bantu?<|end|>
 */
export const USER = "<|user|>", BOT = "<|bot|>", END = "<|end|>";
/**
 * <|konteks|> membuka "kartu konteks" di awal percakapan (nama, topik, suasana, fakta pengguna):
 *   <|konteks|> nama: Sari; tinggal: Medan; topik: masak<|end|><|user|> …
 * Model lama yang dilatih sebelum token ini ada tidak memilikinya; kartu konteks lalu dilewati.
 */
export const KONTEKS = "<|konteks|>";
export const KHUSUS = [USER, BOT, END, KONTEKS];
const RE_KHUSUS = /(<\|user\|>|<\|bot\|>|<\|end\|>|<\|konteks\|>)/;
const RE_PRA = / ?[\p{L}\p{N}]+| ?[^\s\p{L}\p{N}]+|\n/gu;

export function praTokenisasi(teks: string): string[] {
  return teks.match(RE_PRA) ?? [];
}

/** Ambil sampel teks merata (potongan dari seluruh bagian) jika teks lebih panjang dari `maks` karakter. */
function sampelMerata(teks: string, maks: number): string {
  if (teks.length <= maks) return teks;
  const potong = 20000, n = Math.floor(maks / potong), langkah = Math.floor(teks.length / n);
  const bagian: string[] = [];
  for (let i = 0; i < n; i++) bagian.push(teks.slice(i * langkah, i * langkah + potong));
  return bagian.join("\n");
}

/**
 * Melatih BPE sampai ukuran kosakata `ukuranVocab`, atau sampai pasangan terbaik muncul
 * kurang dari `minFreq` kali. Batas frekuensi ini penting: token yang hanya muncul 1–3 kali
 * hampir tidak pernah terlatih, sehingga embedding-nya tetap acak.
 * Untuk teks besar, BPE dilatih pada sampel ≤ 5 juta karakter (hasilnya hampir sama, jauh lebih cepat).
 */
export function latihBPE(teksPenuh: string, ukuranVocab: number, log?: (s: string) => void, minFreq = 10): BPE {
  const teks = sampelMerata(teksPenuh.split(RE_KHUSUS).filter((s) => !KHUSUS.includes(s)).join(" "), 5_000_000);
  // 1. Frekuensi setiap pra-token unik
  const freq = new Map<string, number>();
  for (const w of praTokenisasi(teks)) freq.set(w, (freq.get(w) ?? 0) + 1);
  const kata = [...freq.keys()];
  const jumlah = kata.map((w) => freq.get(w)!);
  const simbol: string[][] = kata.map((w) => [...w]);

  // 2. Kosakata awal = token khusus + semua karakter
  const vocabSet = new Set<string>([UNK, ...KHUSUS]);
  for (const s of simbol) for (const c of s) vocabSet.add(c);
  const vocab = [...vocabSet];

  // 3. Hitung semua pasangan sekali, lalu perbarui secara bertahap setiap merge
  //    pasangan -> jumlah kemunculan, dan pasangan -> kata-kata yang memuatnya
  const kunci = (a: string, b: string) => a + "\u0000" + b;
  const hitung = new Map<string, number>();
  const diKata = new Map<string, Set<number>>();
  const tambahPasangan = (i: number, delta: number) => {
    const s = simbol[i];
    for (let j = 0; j + 1 < s.length; j++) {
      const k = kunci(s[j], s[j + 1]);
      hitung.set(k, (hitung.get(k) ?? 0) + delta * jumlah[i]);
      if (delta > 0) {
        let set = diKata.get(k);
        if (!set) diKata.set(k, (set = new Set()));
        set.add(i);
      }
    }
  };
  for (let i = 0; i < kata.length; i++) tambahPasangan(i, 1);

  const merges: [string, string][] = [];
  while (vocab.length < ukuranVocab) {
    let terbaik = "", maks = minFreq - 1;
    for (const [k, c] of hitung) if (c > maks) { maks = c; terbaik = k; }
    if (!terbaik) break; // tidak ada pasangan yang muncul >= minFreq kali
    const [a, b] = terbaik.split("\u0000");
    const gabung = a + b;
    merges.push([a, b]);
    vocab.push(gabung);
    if (log && merges.length % 250 === 0) log(`merge ${merges.length}: "${a}" + "${b}" -> "${gabung}" (${maks}x)`);

    // Terapkan merge hanya pada kata yang memuat pasangan itu
    for (const i of diKata.get(terbaik) ?? []) {
      const s = simbol[i];
      let ada = false;
      for (let j = 0; j + 1 < s.length; j++) if (s[j] === a && s[j + 1] === b) { ada = true; break; }
      if (!ada) continue;
      tambahPasangan(i, -1);
      const baru: string[] = [];
      for (let j = 0; j < s.length; j++) {
        if (j + 1 < s.length && s[j] === a && s[j + 1] === b) { baru.push(gabung); j++; }
        else baru.push(s[j]);
      }
      simbol[i] = baru;
      tambahPasangan(i, 1);
    }
    hitung.delete(terbaik);
    diKata.delete(terbaik);
  }
  return { vocab, merges };
}

/** Encoder dengan cache per pra-token (kata yang sama tidak di-encode ulang). */
export class Tokenizer {
  readonly vocab: string[];
  private readonly id = new Map<string, number>();
  private readonly peringkat = new Map<string, number>();
  private readonly cache = new Map<string, number[]>();

  constructor(bpe: BPE) {
    this.vocab = bpe.vocab;
    bpe.vocab.forEach((t, i) => this.id.set(t, i));
    bpe.merges.forEach(([a, b], r) => this.peringkat.set(a + "\u0000" + b, r));
  }

  get ukuran(): number {
    return this.vocab.length;
  }

  /** Terapkan merge sesuai peringkat: selalu gabungkan pasangan dengan peringkat terendah dulu. */
  private encodeKata(w: string): number[] {
    const ada = this.cache.get(w);
    if (ada) return ada;
    let s = [...w];
    while (s.length > 1) {
      let rTerbaik = Infinity, posisi = -1;
      for (let j = 0; j + 1 < s.length; j++) {
        const r = this.peringkat.get(s[j] + "\u0000" + s[j + 1]);
        if (r !== undefined && r < rTerbaik) { rTerbaik = r; posisi = j; }
      }
      if (posisi < 0) break;
      s = [...s.slice(0, posisi), s[posisi] + s[posisi + 1], ...s.slice(posisi + 2)];
    }
    const unk = this.id.get(UNK)!;
    const hasil = s.map((t) => this.id.get(t) ?? unk);
    this.cache.set(w, hasil);
    return hasil;
  }

  /** Id token khusus (<|user|>, <|bot|>, <|end|>), atau undefined untuk model lama tanpa token khusus. */
  idToken(t: string): number | undefined {
    return this.id.get(t);
  }

  encode(teks: string): number[] {
    const out: number[] = [];
    for (const bagian of teks.split(RE_KHUSUS)) {
      if (!bagian) continue;
      const khusus = KHUSUS.includes(bagian) ? this.id.get(bagian) : undefined;
      if (khusus !== undefined) { out.push(khusus); continue; }
      for (const w of praTokenisasi(bagian)) out.push(...this.encodeKata(w));
    }
    return out;
  }

  decode(ids: ArrayLike<number>): string {
    let s = "";
    for (let i = 0; i < ids.length; i++) {
      const t = this.vocab[ids[i]];
      s += t === UNK ? "?" : t;
    }
    return s;
  }
}

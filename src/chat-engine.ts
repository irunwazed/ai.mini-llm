/**
 * MESIN PERCAKAPAN: satu logika untuk chat terminal (bun run chat) dan tampilan web (bun run web).
 *
 * Setiap pesan melewati jalur berurutan, dan SETIAP langkah dicatat (proses[]) supaya
 * cara kerjanya bisa dipelajari:
 *   0. "lanjutkan / lebih detail"       -> teks lengkap sumber jawaban sebelumnya
 *   1. Sapaan / basa-basi               -> GPT percakapan (dengan riwayat obrolan + nama dari memori)
 *   2. Pertanyaan aturan                -> penjawab ekstraktif, dasar hukum
 *   3. Pertanyaan tentang diri pengguna -> MEMORI yang disimpan pengguna ("siapa nama saya?")
 *   4. Pesan yang ditujukan ke bot      -> GPT percakapan
 *   5. Pertanyaan fakta umum            -> retrieval Wikipedia, hanya jika yakin
 *   6. Memori (topik lain)              -> catatan pengguna yang cocok
 *   7. Obrolan lain (bukan pertanyaan)  -> GPT percakapan
 *   8. Sisanya                          -> jujur "tidak menemukan" (tidak dikarang)
 *
 * RIWAYAT dipakai dengan dua cara:
 *   - Pertanyaan lanjutan ke aturan/Wikipedia DITULIS ULANG dengan topik sebelumnya
 *     ("kapan dia lahir?" setelah membahas Hö'elün -> "kapan dia lahir? Hö'elün").
 *   - GPT obrolan menerima 2 giliran obrolan terakhir di prompt-nya.
 */
import { existsSync } from "node:fs";
import { join } from "node:path";
import { formatJawaban, jawab, labelSumber, type Jawaban, type Jejak } from "./answer/compose.js";
import { jenisPertanyaan, type Jenis } from "./answer/question-type.js";
import { KONTEKS, Tokenizer } from "./gpt/bpe.js";
import { faktaPengguna, kartuKonteks } from "./gpt/context-card.js";
import { balasDetail, type Giliran, type JejakGPT } from "./gpt/chat.js";
import { gptTerbaru, muatGPT } from "./gpt/checkpoint.js";
import { NAMA_ORANG } from "./gpt/chat-data.js";
import { GPT } from "./gpt/model.js";
import { AMBANG_NIAT, balasanTemplat, keluarga, niatBalasan, niatPesan } from "./chat-intent.js";
import { pahamiMaksud } from "./intent.js";
import { jalankanAgent } from "./reasoning/agent.js";
import { periksaJawaban } from "./reasoning/critic.js";
import { jumlahKataTopik, kataTopik, konteksLebihBaik, mungkinLanjutan, tulisUlang } from "./rag/dialog.js";
import { bacaRAG, type Pembaca } from "./rag/pipeline.js";
import { memoriRelevan } from "./rag/retrieve.js";
import { Memori, PRIBADI } from "./memory.js";
import { normalisasiSlang } from "./text/slang.js";
import { muatModel, type Model } from "./model.js";
import { rincianBM25, teksIndeks } from "./search/bm25.js";
import { keTerm, tokenisasi } from "./text/normalize.js";

export type Jalur = "obrolan" | "aturan" | "wiki" | "memori" | "detail" | "maksud" | "klarifikasi" | "penalaran" | "internet" | "tidak-ditemukan";
/** "agent" dijalankan oleh src/agent/multi-mode.ts (async, bisa ke internet); di proses() diperlakukan sebagai "otomatis". */
export type Mode = "otomatis" | "aturan" | "wiki" | "gpt" | "penalaran" | "agent" | "rag";

export interface LangkahProses {
  tahap: string;       // nama tahap, mis. "Cari di dokumen aturan"
  hasil: string;       // ringkasan keputusan
  ms: number;          // waktu
  data?: unknown;      // rincian (tabel kandidat, token, dll.)
}

export interface SumberTampil { nomor: number; label: string; teks: string }

/** Topik pembicaraan terakhir, untuk memahami pertanyaan lanjutan. */
export interface Topik {
  pertanyaan: string;
  jalur: "aturan" | "wiki" | "internet";
  subjek: string;        // judul artikel (wiki) atau pertanyaan aturan
  sumberLabel: string;
  sumberTeks: string;
}

/** Satu giliran riwayat yang dikirim balik oleh terminal/web. */
export interface GiliranRiwayat { user: string; bot: string; jalur: Jalur }

export interface Respons {
  jalur: Jalur;
  teks: string;                 // jawaban siap tampil (format sama dengan terminal)
  keyakinan?: string;
  sumber: SumberTampil[];
  proses: LangkahProses[];
  gpt?: JejakGPT;
  topik: Topik | null;          // topik baru untuk pertanyaan lanjutan
  label?: string;               // keterangan kecil di bawah jawaban
}

/** rag = jawaban aturan/Wikipedia dibaca ulang oleh GPT pembaca (mode "rag" otomatis menyalakannya). */
export interface OpsiProses { riwayat?: GiliranRiwayat[]; topik?: Topik | null; mode?: Mode; suhu?: number; rag?: boolean }

export const SAPAAN =
  /^(halo+|hai+|hi|hallo|helo|hey|permisi|assalamu|asalamu|selamat (pagi|siang|sore|malam)|(met )?(pagi|siang|sore|malam)\b|apa kabar|(gimana|bagaimana) kabar|terima ?kasih|makasih|thanks|trims|dadah|bye|sampai (jumpa|nanti)|selamat tinggal|kamu siapa|siapa (kamu|namamu)|nama kamu|kamu (bisa apa|robot|manusia)|oke?\b|sip\b|siap\b|tes\b|test\b|ping\b|maaf\b|keren|mantap|nama (saya|aku)|namaku|panggil (aku|saya)|perkenalkan)/i;
/** ", Budi" / ", Siti" dst.: nama dari data latih percakapan yang muncul di balasan GPT. */
const RE_NAMA_LATIH = new RegExp(`,\\s*(${NAMA_ORANG.join("|")})\\b`);
/** Pesan yang ditujukan kepada bot itu sendiri. */
export const KE_BOT = /\b(kamu|kau|engkau|dirimu|kalian)\b|\b(nama|kabar|pencipta|pembuat|umur|usia|tugas|fungsi|kemampuan|asal|hobi|pendapat)mu\b/i;
/** Pesan berbentuk pertanyaan fakta. */
export const TANYA_FAKTA = /\?|\b(apa|apakah|siapa|berapa|kapan|bagaimana|mengapa|kenapa|dimana|di mana|jelaskan|sebutkan)\b/i;
/** "kalau cuti besar?", "bagaimana dengan bakteri?" -> pertanyaan lanjutan dengan subjek BARU. */
const LANJUTAN = /^(kalau|kalo|lalu|terus|trus|bagaimana dengan|gimana dengan|untuk|dan|apakah juga)\b\s*/i;
/** "kapan DIA lahir?", "apa fungsiNYA?" -> merujuk subjek sebelumnya. ("itu/ini" tidak: "apa itu X" bukan rujukan) */
const RUJUK = /\b(dia|ia|beliau|mereka|tersebut)\b|\w{3,}nya\b/i;
/** "lanjutkan", "lebih detail" -> tampilkan teks lengkap sumber sebelumnya. */
/** Jawaban panjang (aturan/Wikipedia/internet) diringkas jadi kalimat pertama ≤ 20 kata untuk riwayat GPT. */
function ringkasBalasan(teks: string, jalur: Jalur): string {
  if (jalur === "obrolan" || jalur === "maksud") return teks;
  const baris = teks.split("\n").map((x) => x.trim()).find((x) => x && !/^(Saya belum|Menurut|Jawaban singkat)/.test(x)) ?? teks;
  const kalimat = baris.replace(/\[\d+\]/g, "").split(/(?<=[.!?])\s/)[0];
  const kata = kalimat.split(/\s+/);
  return kata.length > 20 ? kata.slice(0, 20).join(" ") + " …" : kalimat;
}

/** Ringkasan satu hasil pencarian untuk tabel perbandingan "tanpa konteks vs berkonteks". */
function ringkasJawab(j: Jawaban) {
  const h = j.hasil[0];
  return {
    keyakinan: j.keyakinan, jenis: j.jenis, pasal: h ? `${h.potongan.file} · ${h.potongan.pasal}` : "-",
    cakupan: Number((h?.cakupan ?? 0).toFixed(2)), kekuatan: Number((h?.kekuatan ?? 0).toFixed(2)),
    kalimat: j.utama[0]?.slice(0, 200) ?? "",
  };
}
const MINTA_DETAIL = /^(lanjut(kan)?|lebih (lanjut|detail|lengkap|rinci)|jelaskan lebih|selengkapnya|detail(nya)?|teks lengkap(nya)?|baca selengkapnya)\b/i;

interface ModelGPT { nama: string; model: GPT; tok: Tokenizer; hurufKecil: boolean; langkah: number; lossVal: number }

export class MesinChat {
  readonly aturan: Model;
  readonly wiki: Model | null;
  gpt: ModelGPT | null = null;
  readonly memori: Memori;
  pembaca: Pembaca | null = null;
  /** true = model dipilih otomatis (terbaru), false = dipilih manual (--chat-gpt / panel web) */
  gptOtomatis = true;
  pembacaOtomatis = true;
  private readonly k: number;

  constructor(opsi: { folderModel?: string; folderWiki?: string; chatGpt?: string; k?: number; fileMemori?: string; pembaca?: string } = {}) {
    this.aturan = muatModel(opsi.folderModel ?? "model");
    const fw = opsi.folderWiki ?? "model-wiki";
    this.wiki = existsSync(join(fw, "istilah.json")) ? muatModel(fw) : null;
    this.memori = new Memori(opsi.fileMemori ?? "data/memory.json");
    this.pakaiGPT(opsi.chatGpt ?? null, false);
    this.pakaiPembaca(opsi.pembaca ?? null);
    this.k = opsi.k ?? 3;
  }

  /**
   * Ganti GPT obrolan saat berjalan. null = model obrolan terbaru.
   * `periksaMode` (bawaan true) menolak model yang bukan hasil fine-tune obrolan; dimatikan untuk --chat-gpt (perilaku lama).
   */
  pakaiGPT(nama: string | null, periksaMode = true): void {
    const n = nama ?? gptTerbaru("chat");
    if (!n) { this.gpt = null; this.gptOtomatis = true; return; }
    const { meta, P } = muatGPT(n);
    if (periksaMode && (meta.mode ?? "teks") !== "chat") throw new Error(`"${n}" bukan model obrolan (mode ${meta.mode ?? "teks"})`);
    this.gpt = { nama: n, model: new GPT(meta.config, P), tok: new Tokenizer(meta.bpe), hurufKecil: meta.hurufKecil, langkah: meta.langkah, lossVal: meta.lossValTerbaik };
    this.gptOtomatis = nama === null;
  }

  /** Ganti GPT pembaca (RAG) saat berjalan. null = pembaca terbaru. */
  pakaiPembaca(nama: string | null): void {
    const n = nama ?? gptTerbaru("pembaca");
    if (!n) { this.pembaca = null; this.pembacaOtomatis = true; return; }
    const { meta, P } = muatGPT(n);
    if (meta.mode !== "pembaca") throw new Error(`"${n}" bukan model pembaca (mode ${meta.mode ?? "teks"})`);
    this.pembaca = { nama: n, model: new GPT(meta.config, P), tok: new Tokenizer(meta.bpe), hurufKecil: meta.hurufKecil, ambang: meta.pembaca?.ambang ?? 0.5, f1: meta.pembaca?.f1 ?? 0 };
    this.pembacaOtomatis = nama === null;
  }

  info() {
    return {
      aturan: { pasal: this.aturan.potongan.length, dokumen: new Set(this.aturan.potongan.map((p) => p.file)).size },
      wiki: this.wiki ? { potongan: this.wiki.potongan.length, artikel: new Set(this.wiki.potongan.map((p) => p.file)).size } : null,
      gpt: this.gpt ? { nama: this.gpt.nama, parameter: this.gpt.model.ukuran, lapisan: this.gpt.model.c.layers, dim: this.gpt.model.c.dim, konteks: this.gpt.model.c.context, langkah: this.gpt.langkah, lossVal: this.gpt.lossVal } : null,
      memori: this.memori.semua().length,
      pembaca: this.pembaca ? { nama: this.pembaca.nama, ambang: this.pembaca.ambang, f1: this.pembaca.f1 } : null,
      gptOtomatis: this.gptOtomatis, pembacaOtomatis: this.pembacaOtomatis,
    };
  }

  /** Proses satu pesan dengan riwayat percakapan, topik sebelumnya, dan memori. */
  proses(pesanAsli: string, opsi: OpsiProses = {}): Respons {
    const riwayat = opsi.riwayat ?? [], topik = opsi.topik ?? null, suhu = opsi.suhu ?? 0.4;
    const mode = !opsi.mode || opsi.mode === "agent" || opsi.mode === "rag" ? "otomatis" : opsi.mode;
    const rag = opsi.mode === "rag" || opsi.rag === true;
    const proses: LangkahProses[] = [];
    const ukur = <T>(tahap: string, f: () => T, ringkas: (x: T) => string, data?: (x: T) => unknown): T => {
      const t0 = performance.now();
      const x = f();
      proses.push({ tahap, hasil: ringkas(x), ms: performance.now() - t0, data: data?.(x) });
      return x;
    };

    // 0a. Bahasa santai -> baku ("gimana kalo gak" -> "bagaimana kalau tidak")
    const slang = normalisasiSlang(pesanAsli);
    const pesan = slang.teks;
    if (slang.ganti.length)
      proses.push({ tahap: "Normalisasi bahasa santai", hasil: `"${pesanAsli}" -> "${pesan}"`, ms: 0, data: { ganti: slang.ganti } });

    // 0b. Tokenisasi + konteks yang tersedia
    ukur("Tokenisasi pesan", () => ({ token: tokenisasi(pesan), term: keTerm(pesan) }),
      (x) => `${x.token.length} kata -> ${x.term.length} term (stopword dibuang, ditambah kata dasar "~")`, (x) => x);
    // Nama pengguna: dari memori (disimpan manual), atau dari perkenalan di percakapan ini
    const namaMemori = this.memori.nama();
    const namaSesi = riwayat.map((g) => pahamiMaksud(normalisasiSlang(g.user).teks)).filter((m) => m?.jenis === "perkenalan").map((m) => (m as { nama: string }).nama).at(-1) ?? null;
    const namaPengguna = namaMemori ?? namaSesi;
    proses.push({
      tahap: "Riwayat & memori",
      hasil: `${riwayat.length} giliran sebelumnya; topik terakhir: ${topik ? `"${topik.subjek}" (${topik.jalur})` : "-"}; memori: ${this.memori.semua().length} catatan${namaMemori ? `, nama dari memori: ${namaMemori}` : ""}${!namaMemori && namaSesi ? `, nama dari percakapan ini: ${namaSesi}` : ""}`,
      ms: 0,
      data: { riwayat: riwayat.slice(-4).map((g) => ({ user: g.user, bot: g.bot.split("\n")[0].slice(0, 120), jalur: g.jalur })), topik },
    });

    // 0c. Maksud yang harus selalu benar: perkenalan & pertanyaan tentang diri pengguna
    const maksud = mode === "otomatis" || mode === "gpt" ? pahamiMaksud(pesan) : null;
    if (maksud) {
      let teks: string, label: string;
      if (maksud.jenis === "perkenalan") {
        const n = maksud.nama;
        teks = namaMemori && namaMemori.toLowerCase() === n.toLowerCase()
          ? `Senang bertemu lagi, ${n}! Ada yang bisa saya bantu?`
          : `Senang berkenalan dengan Anda, ${n}! Ada yang bisa saya bantu?${namaMemori ? `\nSebelumnya di memori tercatat nama ${namaMemori}.` : ""}\nAgar saya tetap ingat nama Anda di percakapan berikutnya, klik "simpan ke memori" pada pesan Anda atau ketik :ingat nama saya ${n}`;
        label = "dijawab dengan aturan pemahaman maksud (perkenalan), bukan GPT";
      } else if (namaMemori) {
        teks = `Nama Anda ${namaMemori}.`;
        label = "dari memori yang Anda simpan";
      } else if (namaSesi) {
        teks = `Anda ${namaSesi}, sesuai perkenalan di percakapan ini.\nSaya belum menyimpannya di memori; klik "simpan ke memori" agar saya ingat lain kali.`;
        label = "dari riwayat percakapan ini";
      } else {
        teks = "Saya belum tahu nama Anda. Perkenalkan diri, misalnya: nama saya Budi.";
        label = "dijawab dengan aturan pemahaman maksud, bukan GPT";
      }
      proses.push({ tahap: "Pemahaman maksud", hasil: `maksud: ${maksud.jenis}${maksud.jenis === "perkenalan" ? ` (nama: ${maksud.nama})` : ""} -> jawaban pola tetap (lebih andal daripada GPT kecil)`, ms: 0 });
      return { jalur: "maksud", teks, sumber: [], proses, topik, label, keyakinan: "tinggi" };
    }

    const obrolan = (alasan: string): Respons | null => {
      if (!this.gpt) return null;
      const g = this.gpt;
      const konteks: Giliran[] = [];
      let kartu: string | null = null;
      if (g.tok.idToken(KONTEKS) !== undefined) {
        // Model yang dilatih dengan kartu konteks: kartu fakta pengguna (nama, tinggal, pekerjaan, …) dari
        // memori + SEMUA pesan sesi ini, lalu hingga 6 giliran terakhir dari jalur apa pun (jawaban panjang
        // diringkas). Giliran lama yang tidak muat dipotong oleh balasDetail; kartu tidak pernah dipotong.
        const fakta = faktaPengguna([...riwayat.map((x) => normalisasiSlang(x.user).teks), pesan], { namaMemori, catatanMemori: this.memori.semua().map((c) => c.teks) });
        kartu = kartuKonteks(fakta);
        konteks.push(...riwayat.slice(-6).map((x) => ({ user: x.user, bot: ringkasBalasan(x.bot, x.jalur) })));
        proses.push({ tahap: "Kartu konteks GPT", hasil: kartu ? `<|konteks|> ${kartu}` : "belum ada fakta tentang pengguna", ms: 0, data: { kartu, fakta, giliran: konteks.length } });
      } else {
        // Model lama (tanpa token <|konteks|>): perkenalan nama dari memori + 2 giliran obrolan terakhir
        if (namaPengguna) konteks.push({ user: `nama saya ${namaPengguna}`, bot: `Senang berkenalan dengan Anda, ${namaPengguna}! Ada yang bisa saya bantu?` });
        konteks.push(...riwayat.filter((x) => x.jalur === "obrolan").slice(-2).map((x) => ({ user: x.user, bot: x.bot })));
      }
      proses.push({ tahap: "Pilih jalur", hasil: alasan, ms: 0 });

      // LOOP BEST-OF-N: buat sampai 3 balasan (suhu makin tinggi), periksa niatnya, pakai yang cocok.
      const niat = niatPesan(pesan);
      const target = niat && niat.skor >= AMBANG_NIAT ? keluarga(niat.niat) : null;
      proses.push({
        tahap: "Kenali niat obrolan",
        hasil: niat ? `niat "${niat.niat}" (kemiripan ${niat.skor.toFixed(2)} dengan contoh "${niat.contoh}")${target ? "" : " — terlalu lemah, balasan tidak diperiksa"}` : "niat tidak dikenali",
        ms: 0,
      });
      const percobaan: { suhu: number; teks: string; niat: string | null; skor: number; diterima: boolean }[] = [];
      let r: ReturnType<typeof balasDetail> | null = null, terakhir: ReturnType<typeof balasDetail> | null = null;
      const t0 = performance.now();
      for (let i = 0; i < 3 && !r; i++) {
        const s = Math.min(1.5, suhu + 0.2 * i);
        const x = balasDetail(g.model, g.tok, konteks, pesan, { suhu: s, seed: 1000 + 7 * i + Math.floor(Math.random() * 1e6), simulasi: true, kartu }, g.hurufKecil);
        const nb = niatBalasan(x.teks);
        const diterima = !target || (!!nb && nb.skor >= 0.25 && keluarga(nb.niat) === target);
        percobaan.push({ suhu: s, teks: x.teks, niat: nb?.niat ?? null, skor: nb?.skor ?? 0, diterima });
        terakhir = x;
        if (diterima) r = x;
      }
      let teks: string, templat = false;
      if (r) teks = r.teks;
      else { teks = balasanTemplat(niat!.niat); templat = true; r = terakhir!; }
      proses.push({
        tahap: `GPT "${g.nama}" membalas`,
        hasil: `${percobaan.length} percobaan; ${templat ? `semua ditolak -> balasan contoh niat "${niat!.niat}"` : `diterima percobaan ke-${percobaan.length}`} (${konteks.length} giliran konteks${namaPengguna ? ", termasuk nama" : ""})`,
        ms: performance.now() - t0,
        data: { percobaan, target },
      });
      // Model sekecil ini belum andal MENYALIN nama dari prompt; kadang ia menulis nama lain dari
      // data latihnya ("Sama-sama, Budi!"). Nama seperti itu dikoreksi secara deterministik.
      const salah = teks.match(RE_NAMA_LATIH);
      if (salah && salah[1] !== namaPengguna) {
        teks = namaPengguna ? teks.replace(RE_NAMA_LATIH, `, ${namaPengguna}`) : teks.replace(RE_NAMA_LATIH, "");
        proses.push({
          tahap: "Koreksi nama",
          hasil: namaPengguna
            ? `Model menulis nama "${salah[1]}" (dari data latih), diganti dengan nama dari memori "${namaPengguna}". Model kecil belum andal menyalin nama.`
            : `Model menulis nama "${salah[1]}" padahal nama Anda belum diketahui, jadi nama itu dihapus.`,
          ms: 0,
        });
      }
      return {
        jalur: "obrolan", teks, sumber: [], proses, gpt: r.jejak, topik,
        label: templat ? `balasan contoh untuk niat "${niat!.niat}" (balasan GPT tidak lolos pemeriksaan)` : `obrolan dari model GPT "${g.nama}", bukan dasar hukum`,
      };
    };
    const niatKuat = (() => { const n = niatPesan(pesan); return !!n && n.skor >= AMBANG_NIAT; })();

    // 0d. Soal hitung / logika -> agent penalaran (loop Pikir -> Aksi -> Hasil -> Cek)
    if (mode === "otomatis" || mode === "penalaran") {
      const a = ukur("Agent penalaran", () => jalankanAgent(pesan),
        (x) => x ? (x.terjawab ? `terjawab dengan alat "${x.alat}" setelah ${x.langkah.length} langkah` : `dikenali sebagai soal penalaran, tetapi ${x.langkah.length} alat gagal -> lanjut ke jalur lain`) : "bukan soal hitung/logika yang dikenali",
        (x) => (x ? { langkah: x.langkah } : undefined));
      if (a?.terjawab) {
        proses.push({ tahap: "Pilih jalur", hasil: "Jawaban dari agent penalaran", ms: 0 });
        return { jalur: "penalaran", teks: a.teks, sumber: [], proses, topik, keyakinan: "tinggi", label: `dijawab agent penalaran (${a.alat})` };
      }
      if (mode === "penalaran") {
        return { jalur: "tidak-ditemukan", teks: "Maaf, soal ini belum bisa saya kenali sebagai soal hitung atau logika.", sumber: [], proses, topik };
      }
    }

    // 0. Minta detail dari jawaban sebelumnya
    if (topik?.sumberTeks && MINTA_DETAIL.test(pesan)) {
      proses.push({ tahap: "Pilih jalur", hasil: `Permintaan detail -> teks lengkap sumber jawaban sebelumnya (${topik.sumberLabel})`, ms: 0 });
      return {
        jalur: "detail", topik, proses, keyakinan: "tinggi",
        teks: `Berikut teks lengkap sumbernya [1]:\n\n${topik.sumberTeks}`,
        sumber: [{ nomor: 1, label: topik.sumberLabel, teks: topik.sumberTeks }],
      };
    }

    // 1. Sapaan
    if (mode === "gpt") { const r = obrolan("Mode: hanya GPT"); if (r) return r; }
    if (mode === "otomatis") {
      const sapaan = SAPAAN.test(pesan);
      proses.push({ tahap: "Cek sapaan", hasil: sapaan ? "Ya, pesan berupa sapaan/basa-basi/perkenalan" : "Bukan sapaan", ms: 0 });
      if (sapaan) { const r = obrolan("Sapaan -> GPT percakapan"); if (r) return r; }
    }

    // Pertanyaan yang merujuk subjek pembahasan Wikipedia sebelumnya ("kapan DIA lahir?") ditulis ulang
    // SEBELUM pencarian apa pun, supaya dokumen aturan juga tidak salah menangkap "lahir" -> "melahirkan".
    let kueri = pesan, jenisWarisan: Jenis | undefined, subjek: string | undefined;
    if (topik?.jalur === "wiki" || topik?.jalur === "internet") {
      if (LANJUTAN.test(pesan)) { kueri = pesan.replace(LANJUTAN, ""); jenisWarisan = jenisPertanyaan(topik.pertanyaan); }
      else if (RUJUK.test(pesan)) subjek = topik.subjek;
      if (kueri !== pesan || subjek) proses.push({
        tahap: "Tulis ulang pertanyaan lanjutan",
        hasil: subjek
          ? `"${pesan}" + subjek sebelumnya "${subjek}" (subjek dipakai untuk mencari artikel; kalimat dipilih dari kata yang ditanya)`
          : `"${pesan}" -> "${kueri}" (subjek baru, jenis pertanyaan diwarisi: ${jenisWarisan})`, ms: 0,
      });
    }

    // 2. Aturan (pertanyaan lanjutan aturan membawa pertanyaan sebelumnya sebagai konteks)
    let lanjutanAturan = topik?.jalur === "aturan" && (LANJUTAN.test(pesan) || (keTerm(pesan).length <= 2 && jenisPertanyaan(pesan) === "umum"));
    // RAG: catatan memori pribadi yang relevan ("saya PNS golongan III") ikut menjadi konteks
    const memoriKonteks = rag && PRIBADI.test(pesan) ? memoriRelevan(this.memori.semua(), pesan, true) : [];
    let hasil: Jawaban | null = null, sumberModel = this.aturan;
    if (mode === "otomatis" || mode === "aturan") {
      hasil = ukur("Cari di dokumen aturan (BM25)", () => jawab(this.aturan, kueri, { k: this.k, konteks: lanjutanAturan ? topik!.pertanyaan : "", subjek }),
        (x) => `keyakinan ${x.keyakinan}${lanjutanAturan ? ` (lanjutan dari: "${topik!.pertanyaan}")` : ""}`, (x) => this.ringkasCari(this.aturan, x));
      // LOOP KONTEKS: pertanyaan pendek / berkata ganti ("syaratnya?", "kalau yang itu?") dicari juga
      // dengan konteks pertanyaan sebelumnya (atau catatan memori); dipakai yang hasil pencariannya lebih kuat.
      const calon: { kueri: string; konteks: string; label: string; lanjutan: boolean }[] = [];
      if (topik?.jalur === "aturan" && (lanjutanAturan || mungkinLanjutan(pesan))) {
        if (!lanjutanAturan) calon.push({ kueri, konteks: topik.pertanyaan, label: `konteks pertanyaan sebelumnya "${topik.pertanyaan}"`, lanjutan: true });
        // elipsis: "syaratnya?" -> "syarat cuti besar?" (kata topik pertanyaan sebelumnya mengisi bagian yang hilang)
        const topikLama = kataTopik(topik.pertanyaan);
        const ulang = topikLama ? tulisUlang(kueri.replace(LANJUTAN, ""), topikLama) : kueri;
        if (ulang !== kueri && jumlahKataTopik(kueri.replace(LANJUTAN, "")) === 0) calon.push({ kueri: ulang, konteks: topik.pertanyaan, label: `pertanyaan ditulis ulang "${ulang}"`, lanjutan: true });
      }
      if (memoriKonteks.length && !lanjutanAturan) {
        // kata topik catatan ("saya sedang sakit" -> "sakit") digabung ke pertanyaan, supaya ikut dihitung cakupannya
        const tambahan = memoriKonteks.map((c) => kataTopik(c.teks)).filter(Boolean).join(" ");
        if (tambahan) calon.push({ kueri: `${kueri.replace(/[?.!]+$/, "")} ${tambahan}?`, konteks: "", label: `catatan memori ${memoriKonteks.map((c) => `#${c.id}`).join(", ")} ("${tambahan}" ditambahkan ke pertanyaan)`, lanjutan: false });
      }
      for (const c of calon) {
        const lain = jawab(this.aturan, c.kueri, { k: this.k, konteks: c.konteks, subjek });
        const menang = konteksLebihBaik(lain, hasil!);
        proses.push({
          tahap: "Pahami konteks percakapan",
          hasil: `dicari ulang dengan ${c.label}: keyakinan ${lain.keyakinan} (cakupan ${Math.round(100 * (lain.hasil[0]?.cakupan ?? 0))}%) vs sebelumnya ${hasil!.keyakinan} (${Math.round(100 * (hasil!.hasil[0]?.cakupan ?? 0))}%) -> ${menang ? "dipakai" : "diabaikan (topik baru / tidak lebih baik)"}`,
          ms: 0,
          data: {
            alasan: c.label, menang,
            pilihan: [
              { nama: "tanpa konteks", kueri, konteks: "", ...ringkasJawab(hasil!) },
              { nama: "berkonteks", kueri: c.kueri, konteks: c.konteks, ...ringkasJawab(lain) },
            ],
          },
        });
        if (menang) { hasil = lain; kueri = c.kueri; if (c.lanjutan) lanjutanAturan = true; }
      }
    }
    const tidakAda = () => !hasil || hasil.keyakinan === "rendah";

    // 2b. Pertanyaan terlalu umum yang cocok dengan beberapa pasal berbeda -> tanya balik
    if (hasil && !tidakAda() && !lanjutanAturan) {
      const pilihan = this.pilihanKlarifikasi(hasil);
      if (pilihan) {
        proses.push({ tahap: "Pemeriksa relevansi", hasil: `pertanyaan ambigu: ${pilihan.length} pasal sama kuat dengan topik berbeda -> tanya balik`, ms: 0 });
        return {
          jalur: "klarifikasi", keyakinan: "sedang", proses, sumber: [],
          teks: `Maksud Anda yang mana?\n${pilihan.map((p) => `• ${p.frasa} (${p.pasal})`).join("\n")}\nSebutkan salah satunya, misalnya "${pilihan[0].frasa}".`,
          topik: { pertanyaan: pesan, jalur: "aturan", subjek: pesan, sumberLabel: "", sumberTeks: "" },
          label: "pertanyaan balik karena pertanyaan cocok dengan beberapa aturan",
        };
      }
    }
    // 2c. Pemeriksa relevansi: jawaban aturan yang tidak menjawab pertanyaan ditolak, lalu coba jalur lain
    const periksa = (label: string) => {
      if (!hasil || tidakAda()) return;
      const cek = periksaJawaban(hasil);
      proses.push({ tahap: "Pemeriksa relevansi", hasil: cek.lolos ? `${label}: lolos` : `${label}: DITOLAK — ${cek.alasan.join("; ")} -> coba jalur berikutnya`, ms: 0 });
      if (!cek.lolos) hasil = null;
    };
    periksa("jawaban dokumen aturan");

    // 3. Pertanyaan pribadi -> memori pengguna
    if (mode === "otomatis" && tidakAda() && PRIBADI.test(pesan)) {
      const r = this.dariMemori(pesan, proses, "Pertanyaan tentang diri pengguna -> cari di memori");
      if (r) return { ...r, topik };
    }

    // 4. Pesan untuk bot
    if (mode === "otomatis" && tidakAda() && KE_BOT.test(pesan)) {
      const r = obrolan("Tidak ada aturan yang cocok, dan pesan ditujukan kepada bot -> GPT percakapan");
      if (r) return r;
    }

    // 4b. Pesan yang jelas berupa obrolan ("aku bosan", "boleh minta tolong?") -> GPT, jangan dicari di Wikipedia
    if (mode === "otomatis" && tidakAda() && niatKuat && !subjek) {
      const r = obrolan("Tidak ada aturan yang cocok, dan pesan sangat mirip contoh obrolan -> GPT percakapan");
      if (r) return r;
    }

    // 5. Wikipedia (pertanyaan lanjutan ditulis ulang dengan subjek sebelumnya)
    if ((mode === "otomatis" || mode === "wiki") && this.wiki && tidakAda()) {
      const w = ukur("Cari di Wikipedia (BM25)", () => jawab(this.wiki!, kueri, { k: this.k, jenis: jenisWarisan, subjek }),
        (x) => `keyakinan ${x.keyakinan}${mode === "otomatis" ? " (dipakai hanya jika tinggi)" : ""}`, (x) => this.ringkasCari(this.wiki!, x));
      if (w.keyakinan === "tinggi" || (mode === "wiki" && w.keyakinan !== "rendah")) {
        hasil = w; sumberModel = this.wiki;
        periksa("jawaban Wikipedia");
        if (tidakAda()) sumberModel = this.aturan;
      }
    }

    // 6. Memori untuk topik lain ("kapan rapat tim?")
    if (mode === "otomatis" && tidakAda()) {
      const r = this.dariMemori(pesan, proses, "Tidak ada aturan/Wikipedia yang cocok -> cari di memori");
      if (r) return { ...r, topik };
    }

    // 7. Obrolan lain (hanya jika niatnya dikenali) / 8. tidak ditemukan / belum mengerti
    if (tidakAda()) {
      if (mode === "otomatis" && !TANYA_FAKTA.test(pesan) && niatKuat) {
        const r = obrolan("Tidak ada sumber yang cocok, pesan bukan pertanyaan, dan niat obrolannya dikenali -> GPT percakapan");
        if (r) return r;
      }
      const tanya = TANYA_FAKTA.test(pesan);
      proses.push({ tahap: "Pilih jalur", hasil: tanya ? "Tidak ada sumber yang cukup yakin -> jawab jujur, tidak dikarang" : "Pesan tidak dikenali sebagai pertanyaan maupun obrolan -> minta pengguna menjelaskan", ms: 0 });
      return {
        jalur: "tidak-ditemukan", topik, sumber: [], proses,
        teks: !tanya
          ? "Maaf, saya belum mengerti maksud Anda. Coba tulis dengan kalimat lain, misalnya:\n• pertanyaan aturan: \"berapa hari cuti tahunan?\"\n• pengetahuan umum: \"apa itu virus?\"\n• atau sekadar menyapa: \"halo\""
          : this.wiki
            ? "Maaf, saya tidak menemukan aturan maupun informasi umum tentang hal ini.\nCoba gunakan kata kunci lain, atau tanyakan hal yang berkaitan dengan aturan pemerintah."
            : formatJawaban(this.aturan, jawab(this.aturan, pesan)),
      };
    }

    const j = hasil!;
    const jalur: "aturan" | "wiki" = sumberModel === this.wiki ? "wiki" : "aturan";
    proses.push({ tahap: "Pilih jalur", hasil: jalur === "aturan" ? "Jawaban dari dokumen aturan" : "Jawaban dari Wikipedia", ms: 0 });
    proses.push({ tahap: "Susun jawaban", hasil: `jenis pertanyaan: ${j.jenis}`, ms: 0, data: { catatan: j.jejak.catatan, kalimat: j.jejak.kalimat } });
    const sumber = j.sumber.map((s, i) => ({ nomor: i + 1, label: labelSumber(sumberModel, s).replace(/\n\s*/g, " — "), teks: s.potongan.teks }));
    const utama = j.sumber[0]?.potongan;
    const topikBaru: Topik | null = utama
      ? {
        pertanyaan: jalur === "aturan" && lanjutanAturan ? topik!.pertanyaan : pesan,
        jalur,
        subjek: jalur === "wiki" ? (sumberModel.info[utama.file]?.tentang ?? utama.judul) : (lanjutanAturan ? topik!.subjek : pesan),
        sumberLabel: sumber[0].label,
        sumberTeks: utama.teks,
      }
      : topik;
    let teks = formatJawaban(sumberModel, j), label: string | undefined;

    // RAG: GPT pembaca membaca potongan konteks (termasuk pasal yang dirujuk & catatan memori),
    // lalu jawabannya diverifikasi. Jika lolos, rentang jawaban ditampilkan di depan; penyusun aturan tetap
    // menyusun penjelasan + dasar hukum di bawahnya (jadi tidak pernah lebih buruk dari tanpa RAG).
    if (rag && this.pembaca) {
      const t0 = performance.now();
      const r = bacaRAG(this.pembaca, sumberModel, j, subjek ? `${kueri} ${subjek}` : kueri, { memori: memoriKonteks });
      proses.push({
        tahap: "Ambil konteks (RAG)", ms: 0,
        hasil: `${r.konteks.length} potongan: ${[...new Set(r.konteks.map((k) => k.alasan))].map((a) => `${r.konteks.filter((k) => k.alasan === a).length} ${a}`).join(", ")}`,
        data: { konteks: r.konteks.map((k) => ({ label: k.label, alasan: k.alasan, skor: Number(k.skor.toFixed(2)), cuplikan: k.teks.replace(/\s+/g, " ").slice(0, 160) })) },
      });
      proses.push({
        tahap: `GPT pembaca "${this.pembaca.nama}"`, ms: performance.now() - t0,
        hasil: r.baca ? `rentang "${r.jawaban}" dari ${r.label}, keyakinan ${(100 * r.keyakinan).toFixed(0)}%` : r.alasan.join("; "),
        data: r.baca ? { token: r.baca.token.slice(0, 160), pAwal: r.baca.pAwal.slice(0, 160), pAkhir: r.baca.pAkhir.slice(0, 160), pilih: r.baca.pilih, keyakinan: r.keyakinan, ambang: this.pembaca.ambang, dibaca: r.dibaca } : undefined,
      });
      proses.push({ tahap: "Verifikasi jawaban pembaca", hasil: `${r.diterima ? "DITERIMA" : "DITOLAK -> penyusun aturan"}: ${r.alasan.join("; ")}`, ms: 0 });
      if (r.diterima) {
        teks = `Jawaban singkat: ${r.jawaban} (dibaca GPT pembaca dari ${r.label})\n\n${teks}`;
        label = `RAG: rentang jawaban ditunjuk GPT pembaca (keyakinan ${(100 * r.keyakinan).toFixed(0)}%), penjelasan + sumber dari penyusun aturan`;
      } else label = "RAG: GPT pembaca ragu, jawaban dari penyusun aturan";
      if (memoriKonteks.length) teks += `\n\nCatatan Anda yang dipakai sebagai konteks: ${memoriKonteks.map((c) => `"${c.teks}"`).join("; ")}`;
    }
    return { jalur, keyakinan: j.keyakinan, proses, teks, sumber, topik: topikBaru, label };
  }

  /**
   * Pertanyaan ambigu: hanya satu kata topik ("berapa lama cutinya?"), tetapi ≥ 3 pasal berbeda
   * cocok penuh dengan skor mirip. Pilihan dibuat dari kata yang paling sering mengikuti kata topik
   * di setiap pasal ("cuti" -> "cuti tahunan", "cuti sakit", "cuti melahirkan").
   */
  private pilihanKlarifikasi(j: Jawaban): { frasa: string; pasal: string }[] | null {
    if (!["jumlah", "waktu", "syarat", "yaTidak"].includes(j.jenis)) return null;
    const topikKata = j.jejak.kelompok.filter((g) => g.bobot > 0);
    if (topikKata.length !== 1) return null;
    const kandidat = j.hasil.filter((h) => h.cakupan >= 0.99 && h.skor >= 0.6 * j.hasil[0].skor);
    const akar = topikKata[0].terms.find((t) => t.startsWith("~"))?.slice(1) ?? topikKata[0].kata;
    const pilihan: { frasa: string; pasal: string }[] = [];
    for (const h of kandidat) {
      const hitung = new Map<string, number>();
      for (const m of h.potongan.teks.toLowerCase().matchAll(new RegExp(`\\b${akar}\\w*\\s+([a-z]{3,})`, "g")))
        if (!["yang", "dan", "atau", "dalam", "pada", "untuk", "dari", "sebagaimana", "tersebut", "diberikan", "adalah"].includes(m[1])) hitung.set(m[1], (hitung.get(m[1]) ?? 0) + 1);
      const terbaik = [...hitung].sort((a, b) => b[1] - a[1])[0];
      if (!terbaik) continue;
      const frasa = `${akar} ${terbaik[0]}`;
      if (!pilihan.some((p) => p.frasa === frasa)) pilihan.push({ frasa, pasal: h.potongan.pasal });
    }
    return pilihan.length >= 3 ? pilihan.slice(0, 4) : null;
  }

  dariMemori(pesan: string, proses: LangkahProses[], alasan: string): Omit<Respons, "topik"> | null {
    const t0 = performance.now();
    const m = this.memori.cari(pesan);
    proses.push({
      tahap: "Cari di memori pengguna",
      hasil: m ? `cocok dengan catatan #${m.catatan.id} (cakupan ${Math.round(m.cakupan * 100)}%, kata: ${m.cocok.join(", ")})` : "tidak ada catatan yang cocok (butuh cakupan ≥ 60%)",
      ms: performance.now() - t0,
      data: { catatan: this.memori.semua().length },
    });
    if (!m) return null;
    proses.push({ tahap: "Pilih jalur", hasil: alasan, ms: 0 });
    const label = `Memori #${m.catatan.id}, disimpan ${new Date(m.catatan.waktu).toLocaleString("id-ID")}`;
    return {
      jalur: "memori", keyakinan: "tinggi", proses,
      teks: `${m.jawaban} [1]\n\nSumber:\n  [1] ${label}`,
      sumber: [{ nomor: 1, label, teks: m.catatan.teks }],
      label: "dari memori yang Anda simpan",
    };
  }

  /** Ringkasan pencarian untuk panel proses: term berbobot, kandidat teratas, dan rincian hitungan kandidat #1. */
  private ringkasCari(model: Model, j: Jawaban): { terms: Jejak["terms"]; kandidat: unknown[]; catatan: string[]; rincian?: unknown; cakupan?: unknown } {
    const h = j.hasil[0];
    let rincian, cakupan;
    if (h) {
      const r = rincianBM25(model.indeks, new Map(j.jejak.terms), h.potongan.id);
      rincian = { ...r, baris: r.baris.filter((b) => b.f > 0 || b.bobot >= 1).slice(0, 12), skorAkhir: h.skor, faktor: r.total ? h.skor / r.total : 1 };
      const ada = new Set(keTerm(`${model.info[h.potongan.file]?.tentang ?? ""} ${teksIndeks(h.potongan)}`));
      const lnN = Math.log(1 + model.indeks.N);
      cakupan = {
        lnN,
        kelompok: j.jejak.kelompok.map((g) => ({ kata: g.kata, bobot: g.bobot, cocok: g.terms.some((t) => ada.has(t)) })),
        cakupan: h.cakupan, kekuatan: h.kekuatan,
      };
    }
    return {
      terms: j.jejak.terms.slice(0, 16),
      rincian, cakupan,
      catatan: j.jejak.catatan,
      kandidat: j.hasil.slice(0, 5).map((h) => ({
        sumber: labelSumber(model, { potongan: h.potongan, ayat: new Set() }).split("\n")[0],
        bm25: Number(h.skor.toFixed(2)),
        cakupan: Number(h.cakupan.toFixed(2)),
        kekuatan: Number(h.kekuatan.toFixed(2)),
        cuplikan: h.potongan.teks.replace(/\s+/g, " ").slice(0, 160),
      })),
    };
  }
}

// ------------------------------------------------------------------------------------------
// EKSPOR PERCAKAPAN: teks ringkas yang bisa disalin lalu ditempel untuk diskusi/perbaikan.
// ------------------------------------------------------------------------------------------
export interface EntriEkspor {
  pesan: string; jalur: string; teks: string; keyakinan?: string; label?: string; totalMs?: number;
  proses: { tahap: string; hasil: string }[];
  sumber?: { nomor: number; label: string }[];
  tokenGPT?: { token: string; p: number }[];
}

export function eksporMarkdown(entri: EntriEkspor[], info: ReturnType<MesinChat["info"]>, opsi: { mode?: string; suhu?: number } = {}): string {
  const baris: string[] = [];
  baris.push(`# Ekspor percakapan — ${new Date().toLocaleString("id-ID")}`);
  baris.push(`Model: aturan ${info.aturan.pasal} pasal/${info.aturan.dokumen} dokumen · Wikipedia ${info.wiki ? `${info.wiki.artikel} artikel` : "-"} · GPT ${info.gpt ? `"${info.gpt.nama}" (${info.gpt.parameter} parameter, langkah ${info.gpt.langkah}, loss val ${info.gpt.lossVal.toFixed(3)})` : "-"} · memori ${info.memori} catatan`);
  baris.push(`Mode: ${opsi.mode ?? "otomatis"} · suhu GPT ${opsi.suhu ?? 0.4}`);
  entri.forEach((e, i) => {
    baris.push("", `## ${i + 1}. Anda: ${e.pesan}`);
    baris.push(`Bot [${e.jalur}${e.keyakinan ? `, keyakinan ${e.keyakinan}` : ""}${e.totalMs !== undefined ? `, ${Math.round(e.totalMs)} ms` : ""}]:`);
    baris.push(...e.teks.split("\n").map((l) => "  " + l));
    if (e.label) baris.push(`  (${e.label})`);
    baris.push("Proses:");
    for (const p of e.proses) baris.push(`- ${p.tahap}: ${p.hasil}`);
    if (e.tokenGPT?.length) baris.push(`- Token GPT: ${e.tokenGPT.map((t) => `${JSON.stringify(t.token)} ${Math.round(t.p * 100)}%`).join(", ")}`);
  });
  return baris.join("\n") + "\n";
}

/** Ubah Respons menjadi entri ekspor (tanpa data besar seperti matriks). */
export function keEntriEkspor(pesan: string, r: Respons & { totalMs?: number }): EntriEkspor {
  return {
    pesan, jalur: r.jalur, teks: r.teks, keyakinan: r.keyakinan, label: r.label, totalMs: r.totalMs,
    proses: r.proses.map((p) => ({ tahap: p.tahap, hasil: p.hasil })),
    sumber: r.sumber.map((s) => ({ nomor: s.nomor, label: s.label })),
    tokenGPT: r.gpt?.langkah.map((l) => ({ token: l.token, p: l.p })),
  };
}

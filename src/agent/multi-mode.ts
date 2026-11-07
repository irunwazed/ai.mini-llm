/**
 * MODE AGENT: loop yang memakai SEMUA mode, lalu memilih jawaban yang paling cocok.
 *
 *   Putaran 1 — coba semua mode lokal sebagai "alat":
 *               otomatis, penalaran, dokumen aturan, Wikipedia lokal, GPT obrolan
 *   Nilai     — setiap kandidat diberi skor 0–1 yang bisa dijelaskan:
 *               jalur pasti (maksud/penalaran/memori) tinggi; aturan/Wikipedia = keyakinan × cakupan kata;
 *               GPT tinggi untuk sapaan, rendah untuk pertanyaan fakta; "tidak ditemukan" = 0
 *   Cek       — jika skor terbaik ≥ AMBANG_PUAS, selesai.
 *   Putaran 2+— jika belum puas DAN pesan berupa pertanyaan: cari di internet, berulang dengan
 *               strategi berbeda sampai lolos cek atau habis:
 *                 a. Wikipedia Indonesia (kata kunci) -> paragraf pembuka
 *                 b. artikel Wikipedia lengkap dari hasil teratas
 *                 c. Brave Search API (hanya jika BRAVE_API_KEY diset)
 *   Pilih     — kandidat berskor tertinggi; semua langkah tercatat (Pikir/Aksi/Hasil/Cek).
 */
import { kataPertanyaan, namaDiri, pilihKalimatWeb } from "../answer/web-extract.js";
import { besaran, jenisPertanyaan, satuanDiminta } from "../answer/question-type.js";
import { AMBANG_NIAT, niatPesan } from "../chat-intent.js";
import { KE_BOT, SAPAAN, TANYA_FAKTA, type LangkahProses, type MesinChat, type Mode, type OpsiProses, type Respons } from "../chat-engine.js";
import type { LangkahAgent } from "../reasoning/agent.js";
import { adaKunciWeb, artikelWikipedia, cariWeb, cariWikipedia, type HalamanWeb } from "../search/internet.js";
import { keTerm } from "../text/normalize.js";
import { normalisasiSlang } from "../text/slang.js";

export const AMBANG_PUAS = 0.6;
export interface OpsiAgent extends OpsiProses { internet?: boolean }

interface Kandidat { mode: string; r: Respons; skor: number; alasan: string }

/** Bagian jawaban sebelum daftar sumber. */
const inti = (t: string) => t.split(/\n\s*\n(?=Sumber|Dasar hukum|Keyakinan)/i)[0].replace(/\[\d+\]/g, ""); // "[1]" bukan angka jawaban

/** Persentase kata pertanyaan (berbobot sama) yang muncul di teks jawaban. */
function cakupanKata(pertanyaan: string, teks: string): number {
  const kata = kataPertanyaan(pertanyaan);
  if (!kata.length) return 1;
  const ada = new Set(keTerm(teks));
  return kata.filter((k) => keTerm(k).some((t) => ada.has(t))).length / kata.length;
}

/** Pesan berupa obrolan: sapaan, atau sangat mirip contoh niat obrolan ("boleh minta tolong?"). */
function obrolanJelas(pesan: string): boolean {
  const niat = niatPesan(pesan);
  return SAPAAN.test(pesan) || (!!niat && niat.skor >= AMBANG_NIAT);
}
/** Pertanyaan fakta yang butuh sumber (bukan obrolan, bukan pertanyaan tentang bot). */
export function tanyaFakta(pesan: string): boolean {
  return TANYA_FAKTA.test(pesan) && !KE_BOT.test(pesan) && !obrolanJelas(pesan);
}

function nilai(pesan: string, r: Respons): { skor: number; alasan: string } {
  const tanya = tanyaFakta(pesan);
  switch (r.jalur) {
    case "tidak-ditemukan": return { skor: 0.3, alasan: "jawaban jujur \"tidak tahu\" (lebih baik daripada jawaban ngawur)" };
    case "maksud": return { skor: 1, alasan: "maksud dikenali dengan pola pasti" };
    case "detail": return { skor: 1, alasan: "permintaan detail sumber sebelumnya" };
    case "penalaran": return { skor: 0.95, alasan: "dihitung alat dan lolos cek ulang" };
    case "memori": return { skor: 0.85, alasan: "cocok dengan catatan memori Anda" };
    case "klarifikasi": return { skor: 0.55, alasan: "pertanyaan ambigu -> tanya balik" };
    case "obrolan": {
      if (obrolanJelas(pesan) || (KE_BOT.test(pesan) && TANYA_FAKTA.test(pesan))) return { skor: 0.8, alasan: "pesan berupa obrolan / ditujukan ke bot" };
      if (!tanya) return { skor: 0.25, alasan: "niat obrolan tidak dikenali; balasan GPT bisa ngawur" };
      return { skor: 0.2, alasan: "pertanyaan fakta tidak boleh dijawab karangan GPT" };
    }
    case "aturan": case "wiki": case "internet": {
      // skor = keyakinan × (0,2 + 0,8 × cakupan²): satu kata pertanyaan yang hilang ("lahir", "tinggi")
      // langsung menurunkan skor cukup banyak, karena biasanya justru kata itulah yang ditanyakan.
      const k = r.keyakinan === "tinggi" ? 0.9 : r.keyakinan === "sedang" ? 0.65 : 0.25;
      const isi = inti(r.teks);
      const c = cakupanKata(pesan, isi);
      let skor = k * (0.2 + 0.8 * c * c);
      const alasan = [`keyakinan ${r.keyakinan ?? "-"} × cakupan kata ${Math.round(100 * c)}%`];
      const jenis = jenisPertanyaan(pesan);
      const satuan = satuanDiminta(pesan);
      if (jenis === "jumlah" && !/\d/.test(isi)) { skor *= 0.4; alasan.push(`ditanya "berapa" tetapi tanpa angka`); }
      else if (jenis === "jumlah" && satuan && !besaran(isi).some((b) => satuan.test(b.toLowerCase()))) { skor *= 0.4; alasan.push(`ditanya "berapa" tetapi tanpa satuan yang diminta`); }
      if (jenis === "waktu" && !/\d/.test(isi)) { skor *= 0.5; alasan.push("ditanya waktu tetapi tanpa tanggal/angka"); }
      if (obrolanJelas(pesan)) { skor *= 0.3; alasan.push("pesan berupa obrolan, bukan pertanyaan"); }
      return { skor, alasan: alasan.join("; ") };
    }
  }
}

const ringkas = (t: string) => { const s = inti(t).split("\n")[0]; return s.length > 140 ? s.slice(0, 137) + "…" : s; };

/** Jalankan mode agent. Async karena bisa mencari di internet. */
export async function jalankanModeAgent(mesin: MesinChat, pesanAsli: string, opsi: OpsiAgent = {}): Promise<Respons> {
  const t0 = performance.now();
  const langkah: LangkahAgent[] = [];
  const kandidat: Kandidat[] = [];
  const pesan = normalisasiSlang(pesanAsli).teks;

  // ---- Putaran 1: semua mode lokal ----
  const daftarMode: { mode: Mode; pikir: string }[] = [
    { mode: "otomatis", pikir: "Mulai dari jalur otomatis (urutan bawaan: maksud, sapaan, aturan, Wikipedia, memori)." },
    { mode: "penalaran", pikir: "Mungkin ini soal hitung/logika: coba agent penalaran." },
    { mode: "aturan", pikir: "Cari langsung di dokumen aturan, walau jalur otomatis memilih yang lain." },
    ...(mesin.wiki ? [{ mode: "wiki" as Mode, pikir: "Cari di Wikipedia lokal dengan ambang yang lebih longgar." }] : []),
    ...(mesin.gpt ? [{ mode: "gpt" as Mode, pikir: "Minta GPT obrolan membalas (untuk sapaan/basa-basi)." }] : []),
  ];
  for (const { mode, pikir } of daftarMode) {
    // GPT dilewati jika jalur otomatis sudah memakai GPT, atau jika pesannya pertanyaan fakta (GPT tidak boleh menjawab fakta)
    if (mode === "gpt" && (kandidat.some((k) => k.r.jalur === "obrolan") || tanyaFakta(pesan))) continue;
    const r = mesin.proses(pesanAsli, { ...opsi, mode });
    const { skor, alasan } = nilai(pesan, r);
    const sama = kandidat.find((k) => k.r.jalur === r.jalur && k.r.teks === r.teks);
    langkah.push({
      pikir, aksi: `mode "${mode}"`,
      hasil: `jalur ${r.jalur}: ${ringkas(r.teks)}`,
      cek: sama ? `sama dengan hasil mode "${sama.mode}" -> diabaikan` : `skor ${skor.toFixed(2)} (${alasan})`,
      rincian: r.proses.filter((p) => !["Tokenisasi pesan", "Riwayat & memori"].includes(p.tahap)).map((p) => `${p.tahap}: ${p.hasil}`).slice(-6),
    });
    if (!sama) kandidat.push({ mode, r, skor, alasan });
  }
  let terbaik = kandidat.reduce((a, b) => (b.skor > a.skor ? b : a));

  // ---- Putaran 2+: internet ----
  const tanya = TANYA_FAKTA.test(pesan) || jenisPertanyaan(pesan) !== "umum";
  const bolehInternet = opsi.internet !== false;
  if (terbaik.skor < AMBANG_PUAS && tanya && bolehInternet) {
    const web = await cariDiInternet(pesan, langkah, terbaik.skor, opsi.topik ?? null);
    if (web) {
      const { skor, alasan } = nilai(pesan, web);
      kandidat.push({ mode: "internet", r: web, skor, alasan });
      if (skor > terbaik.skor) terbaik = kandidat.at(-1)!;
    }
  } else if (terbaik.skor < AMBANG_PUAS) {
    langkah.push({
      pikir: `Skor terbaik ${terbaik.skor.toFixed(2)} < ${AMBANG_PUAS}.`,
      aksi: "tidak mencari di internet",
      hasil: !bolehInternet ? "pencarian internet dimatikan" : "pesan bukan pertanyaan, jadi tidak ada yang perlu dicari",
      cek: "-", rincian: [],
    });
  }

  const urut = [...kandidat].sort((a, b) => b.skor - a.skor);
  langkah.push({
    pikir: "Bandingkan semua kandidat dan pilih skor tertinggi.",
    aksi: "pilih jawaban",
    hasil: `dipilih mode "${terbaik.mode}" (jalur ${terbaik.r.jalur})`,
    cek: terbaik.skor >= AMBANG_PUAS ? `skor ${terbaik.skor.toFixed(2)} ≥ ${AMBANG_PUAS} ✓` : `skor ${terbaik.skor.toFixed(2)} masih < ${AMBANG_PUAS}: jawaban terbaik yang ada, belum yakin`,
    rincian: urut.map((k, i) => `${i + 1}. ${k.mode.padEnd(9)} ${k.skor.toFixed(2)}  ${k.r.jalur} — ${k.alasan}`),
  });

  const ringkasan: LangkahProses = {
    tahap: "Agent multi-mode",
    hasil: `${kandidat.length} kandidat dari ${langkah.length - 1} langkah -> dipilih "${terbaik.mode}" (skor ${terbaik.skor.toFixed(2)})`,
    ms: performance.now() - t0,
    data: { langkah, multiMode: true },
  };
  const r = terbaik.r;
  return {
    ...r,
    proses: [ringkasan, ...r.proses],
    label: `${r.label ? r.label + " · " : ""}dipilih agent dari ${kandidat.length} kandidat (skor ${terbaik.skor.toFixed(2)})`,
  };
}

/**
 * Loop pencarian internet: beberapa strategi, berhenti begitu ada kalimat yang lolos cek:
 * ≤ 3 kata kunci -> SEMUA harus ada; lebih banyak -> ≥ 70% kata dan ≥ 50% bobot IDF. Nama diri wajib ada.
 */
async function cariDiInternet(pesan: string, langkah: LangkahAgent[], skorLokal: number, topik: Respons["topik"]): Promise<Respons | null> {
  const kata = kataPertanyaan(pesan);
  const kunci = kata.join(" ");
  if (!kunci) return null;
  const kumpulan: HalamanWeb[] = [];
  const proses: LangkahProses[] = [];
  const nama = namaDiri(pesan);
  const jenis = jenisPertanyaan(pesan);
  const syarat = (kata.length <= 3 ? "semua kata kunci" : "≥ 70% kata kunci") + (jenis === "jumlah" || jenis === "waktu" ? " + angka" : "");
  const cukup = (k: { teks: string; cakupan: number; cocok: string[] }) =>
    k.cakupan >= 0.5 && (kata.length <= 3 ? k.cocok.length === kata.length : k.cocok.length / kata.length >= 0.7) &&
    ((jenis !== "jumlah" && jenis !== "waktu") || /\d/.test(k.teks)); // "berapa"/"kapan" harus dijawab dengan angka
  /** Halaman wiki yang judulnya paling cocok dengan subjek pertanyaan (untuk dibaca lengkap). */
  const halamanUtama = () => {
    const kataQ = new Set(keTerm(pesan));
    const nilaiJudul = (h: HalamanWeb) => { const k = keTerm(h.judul).filter((t) => !t.startsWith("~")); return k.length ? k.filter((t) => kataQ.has(t)).length / k.length : 0; };
    return kumpulan.filter((h) => h.asal === "Wikipedia").reduce<HalamanWeb | null>((a, h) => (!a || nilaiJudul(h) > nilaiJudul(a) ? h : a), null);
  };

  // Ada nama diri ("Soekarno") -> cari artikel tentang subjeknya DULU; kata kunci lengkap ("soekarno lahir")
  // cenderung menemukan orang lain yang bernama sama dan juga "lahir".
  const awal = `Sumber lokal belum cukup (skor terbaik ${skorLokal.toFixed(2)} < ${AMBANG_PUAS}) -> cari di internet.`;
  const subjek = nama.join(" ");
  type Strategi = { pikir: string; aksi: string; jalan: () => Promise<HalamanWeb[]> };
  const lengkap: Strategi = {
    pikir: "Paragraf pembuka belum memuat jawabannya -> baca artikel lengkap yang judulnya paling cocok.",
    aksi: "Wikipedia API: artikel lengkap", jalan: async () => {
      // Judul persis dari frasa berhuruf kapital ("Jakarta", "Laskar Pelangi"); Wikipedia mengikuti pengalihan
      // ("Jakarta" -> "Daerah Khusus Ibukota Jakarta"). Jika tidak ada, pakai hasil pencarian yang judulnya paling cocok.
      const frasa = pesan.match(/(?<=\s)[A-Z][\p{L}'-]+(?:\s+[A-Z][\p{L}'-]+)*/u)?.[0];
      if (frasa) { const a = await artikelWikipedia(frasa).catch(() => []); if (a.length) return a; }
      return halamanUtama() ? artikelWikipedia(halamanUtama()!.judul) : [];
    },
  };
  const kataKunci: Strategi = { pikir: awal, aksi: `Wikipedia API: cari "${kunci}" (paragraf pembuka 5 artikel)`, jalan: () => cariWikipedia(kunci, 5) };
  const strategi: Strategi[] = nama.length && subjek !== kunci
    ? [
      { pikir: `${awal} Ada nama diri (${nama.join(", ")}), jadi cari artikel tentang subjeknya dulu.`, aksi: `Wikipedia API: cari "${subjek}" (paragraf pembuka 3 artikel)`, jalan: () => cariWikipedia(subjek, 3) },
      lengkap,
      { ...kataKunci, pikir: "Artikel subjek belum memuat jawabannya -> cari dengan semua kata kunci." },
    ]
    : [kataKunci, lengkap];
  strategi.push(
    ...(adaKunciWeb() ? [{ pikir: "Wikipedia belum cukup -> cari di web umum.", aksi: `Brave Search API: "${pesan}"`, jalan: () => cariWeb(pesan, 5) }] : []),
  );

  for (const s of strategi) {
    const t0 = performance.now();
    let baru: HalamanWeb[] = [], galat = "";
    try { baru = await s.jalan(); } catch (e) { galat = (e as Error).message; }
    if (s.aksi.endsWith("artikel lengkap") && baru[0]) s.aksi = `Wikipedia API: artikel lengkap "${baru[0].judul}"`;
    for (const h of baru) if (!kumpulan.some((k) => k.url === h.url && k.asal === h.asal)) kumpulan.push(h);
    const pilih = pilihKalimatWeb(pesan, kumpulan, 2, cukup);
    const k0 = pilih.kalimat[0];
    const lolos = !!k0 && cukup(k0);
    proses.push({ tahap: "Cari di internet", hasil: `${s.aksi}: ${galat ? `gagal (${galat})` : `${baru.length} halaman`}`, ms: performance.now() - t0 });
    langkah.push({
      pikir: s.pikir, aksi: s.aksi,
      hasil: galat ? `gagal: ${galat}` : `${baru.length} halaman baru (${baru.map((h) => h.judul).slice(0, 3).join(", ") || "-"}); ${pilih.jumlahKalimat} kalimat dinilai`,
      cek: k0 ? `kalimat terbaik memuat ${Math.round(100 * k0.cakupan)}% kata pertanyaan (${k0.cocok.join(", ")}) ${lolos ? "✓" : `✗ (butuh ${syarat})`}` : "tidak ada kalimat yang memuat kata pertanyaan ✗",
      rincian: pilih.kalimat.map((k) => `[${kumpulan[k.halaman].judul}] skor ${k.skor.toFixed(2)}: ${k.teks.slice(0, 160)}`),
    });
    if (!lolos) continue;

    // Susun jawaban: kutipan kalimat + daftar sumber ber-URL
    const pakai = pilih.kalimat.filter(cukup);
    const nomor = new Map<number, number>();
    for (const k of pakai) if (!nomor.has(k.halaman)) nomor.set(k.halaman, nomor.size + 1);
    const sumber = [...nomor].map(([h, n]) => ({ nomor: n, label: `${kumpulan[h].judul} — ${kumpulan[h].url} (${kumpulan[h].asal})`, teks: kumpulan[h].teks }));
    const teks = [
      pakai.map((k) => `${k.teks} [${nomor.get(k.halaman)}]`).join(" "),
      "",
      "Sumber (internet):",
      ...sumber.map((s) => `  [${s.nomor}] ${s.label}`),
      "",
      "Catatan: kutipan dari internet, bukan dasar hukum. Periksa sumbernya.",
    ].join("\n");
    proses.push({ tahap: "Susun jawaban", hasil: `${pakai.length} kalimat dikutip dari ${sumber.length} halaman (BM25 mini atas kalimat halaman)`, ms: 0 });
    return {
      jalur: "internet", teks, sumber, proses, topik,
      keyakinan: k0.cakupan >= 0.8 ? "tinggi" : "sedang",
      label: "kutipan dari internet (API Wikipedia" + (adaKunciWeb() ? "/Brave" : "") + "), bukan dasar hukum",
    };
  }
  return null;
}

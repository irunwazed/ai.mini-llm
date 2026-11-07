/**
 * MODE AGENT: loop "pahami -> pakai alat -> bandingkan -> perbaiki -> simpulkan".
 *
 *   1. PAHAMI MAKSUD  (intent.ts)   jenis pesan, bentuk jawaban yang diharapkan, kata kunci, nama diri,
 *                                    subjek percakapan sebelumnya, sumber yang paling mungkin
 *   2. PAKAI ALAT     (tools.ts)    ambil aksi teratas dari antrian (alat + kueri); aksi yang sudah
 *                                    pernah dicoba dilewati
 *   3. BANDINGKAN     (compare.ts)  hasil alat vs maksud -> skor + daftar kekurangan
 *   4. PERBAIKI       (di sini)     kekurangan -> aksi baru di DEPAN antrian
 *                                    (cari subjeknya, baca artikel lengkap, pakai sinonim, ganti sumber)
 *                                    ulangi 2–4 sampai cocok, antrian habis, atau batas langkah/waktu
 *   5. SIMPULKAN      (synthesize.ts) SEMUA hasil (pasal, Wikipedia, semua halaman internet, memori)
 *                                    dikumpulkan jadi kolam bukti, kalimatnya dinilai terhadap maksud,
 *                                    lalu agent menyusun jawabannya sendiri + sumber + jejak; jika belum yakin,
 *                                    katakan jujur dan ajukan pertanyaan balik
 * Semua langkah dicatat sebagai Pikir/Aksi/Hasil/Cek untuk panel cara kerja.
 */
import type { LangkahProses, MesinChat, OpsiProses, Respons } from "../chat-engine.js";
import type { LangkahAgent } from "../reasoning/agent.js";
import { padanan } from "../text/synonyms.js";
import { AMBANG_PUAS, bandingkan, inti, type Kekurangan } from "./compare.js";
import { pahamiNiat, type Niat } from "./intent.js";
import { simpulkan, type Percobaan } from "./synthesize.js";
import { alasanDilewati, alatTersedia, jalankanAlat, judulArtikel, type Aksi, type KonteksAlat, type NamaAlat } from "./tools.js";

export { AMBANG_PUAS } from "./compare.js";
export { tanyaFakta } from "./intent.js";
export interface OpsiAgent extends OpsiProses { internet?: boolean; maksLangkah?: number; maksDetik?: number }

/** Jalur yang hasilnya pasti: begitu cocok, tidak perlu mencari lagi. */
const JALUR_PASTI = new Set(["maksud", "detail", "penalaran", "obrolan", "memori", "klarifikasi"]);

/** Rencana awal: urutan alat sesuai jenis maksud. */
function rencanaAwal(niat: Niat, pesanAsli: string): Aksi[] {
  const a = (alat: NamaAlat, kueri: string, pikir: string): Aksi => ({ alat, kueri, pikir });
  const kunci = niat.kataCari.join(" ");
  const subjek = niat.entitas.join(" ");
  const otomatis = a("otomatis", pesanAsli, "Mulai dari jalur bawaan (maksud, sapaan, aturan, Wikipedia, memori).");
  switch (niat.jenis) {
    case "diri": return [otomatis];
    case "penalaran": return [a("penalaran", pesanAsli, "Soal hitung/logika -> agent penalaran."), otomatis];
    case "obrolan": return [otomatis, a("obrolan-gpt", pesanAsli, "Obrolan -> GPT percakapan.")];
    case "pribadi": return [
      a("memori", pesanAsli, "Pertanyaan tentang diri Anda -> cari di catatan memori."),
      a("rag", pesanAsli, "Cari aturan dengan catatan memori sebagai konteks (RAG)."), otomatis,
    ];
    case "aturan": return [
      otomatis, a("aturan", pesanAsli, "Pertanyaan aturan -> cari langsung di dokumen aturan."),
      a("rag", pesanAsli, "Baca ulang dengan GPT pembaca + pasal yang dirujuk (RAG)."),
      a("wiki-lokal", pesanAsli, "Mungkin istilah umum -> Wikipedia lokal."),
      a("wiki-kata-kunci", kunci, "Belum ketemu di sumber lokal -> cari di Wikipedia internet."),
    ];
    case "fakta": return [
      otomatis, a("wiki-lokal", pesanAsli, "Pengetahuan umum -> Wikipedia lokal."),
      ...(subjek ? [a("wiki-subjek", subjek, `Ada nama diri (${subjek}) -> cari artikel tentang subjeknya.`)] : []),
      a("wiki-kata-kunci", kunci, "Cari di Wikipedia internet dengan semua kata kunci."),
      a("wiki-artikel", "", "Baca artikel lengkap yang judulnya paling cocok."),
      a("duckduckgo", kunci, "Coba definisi/entitas dari DuckDuckGo (API resmi)."),
      a("brave", niat.kueri, "Cari di web umum (Brave Search API)."),
    ];
    default: return [otomatis, a("aturan", pesanAsli, "Coba dokumen aturan."), a("wiki-lokal", pesanAsli, "Coba Wikipedia lokal.")];
  }
}

/** Ubah kekurangan menjadi aksi perbaikan (ditaruh di depan antrian). */
function perbaikan(niat: Niat, kurang: Kekurangan[], ctx: KonteksAlat, pesanAsli: string): Aksi[] {
  const aksi: Aksi[] = [];
  // urutan perbaikan: nama diri dulu (cari artikel subjeknya), lalu bentuk jawaban, sumber, baru kata kunci
  const prioritas: Record<string, number> = { entitas: 0, bentuk: 1, sumber: 2, kata: 3, kosong: 4, ambigu: 5 };
  for (const k of [...kurang].sort((a, b) => prioritas[a.jenis] - prioritas[b.jenis])) {
    if (k.jenis === "entitas") {
      const s = k.kata!.join(" ");
      aksi.push({ alat: "wiki-subjek", kueri: s, pikir: `Nama "${s}" tidak disebut -> cari artikel tentang ${s}.` });
      aksi.push({ alat: "wiki-artikel", kueri: "", pikir: `Baca artikel lengkap tentang ${s}.` });
    } else if (k.jenis === "bentuk") {
      if (ctx.mesin.pembaca && niat.jenis === "aturan") aksi.push({ alat: "rag", kueri: pesanAsli, pikir: `${k.rincian} -> minta GPT pembaca menunjuk jawaban di pasal.` });
      aksi.push({ alat: "wiki-artikel", kueri: "", pikir: `${k.rincian} -> paragraf pembuka belum cukup, baca artikel lengkap.` });
    } else if (k.jenis === "kata") {
      // kata yang ditanya tidak ada -> coba padanan katanya ("hamil" -> "melahirkan")
      const ganti = k.kata!.map((w) => [w, padanan(w)[0]] as const).filter(([, p]) => p);
      if (ganti.length) {
        let pesan = niat.pesan, kunci = niat.kataCari.join(" ");
        for (const [w, p] of ganti) { pesan = pesan.replace(new RegExp(`\\b${w}\\b`, "i"), p!); kunci = kunci.replace(w, p!); }
        const tujuan: NamaAlat = niat.jenis === "aturan" || niat.domain === "aturan" ? "aturan" : "wiki-lokal";
        aksi.push({ alat: tujuan, kueri: pesan, pikir: `Kata "${ganti.map(([w]) => w).join(", ")}" tidak ditemukan -> coba padanannya (${ganti.map(([, p]) => p).join(", ")}).` });
        aksi.push({ alat: "wiki-kata-kunci", kueri: kunci, pikir: "Cari di internet dengan padanan kata." });
      } else aksi.push({ alat: "wiki-kata-kunci", kueri: niat.kataCari.join(" "), pikir: `Kata "${k.kata!.join(", ")}" tidak ditemukan di sumber lokal -> cari di internet.` });
    } else if (k.jenis === "sumber") {
      if (niat.jenis === "aturan") aksi.push({ alat: "aturan", kueri: pesanAsli, pikir: "Pertanyaan aturan -> utamakan dokumen aturan." });
      else aksi.push({ alat: "wiki-lokal", kueri: pesanAsli, pikir: "Butuh sumber fakta, bukan karangan GPT -> Wikipedia lokal." });
    }
  }
  if (niat.subjekLanjutan && niat.kueri !== niat.pesan) aksi.unshift({ alat: "wiki-kata-kunci", kueri: niat.kataCari.join(" "), pikir: `Pertanyaan lanjutan -> cari dengan subjek "${niat.subjekLanjutan}".` });
  return aksi;
}

/** Rincian lengkap satu alat untuk panel cara kerja (jawaban apa adanya, penilaian, sumber, proses di dalam alat). */
function detailAlat(aksi: Aksi, r: Respons, nilai: ReturnType<typeof bandingkan>, lanjut: Aksi[], ms: number, samaDengan: string | null) {
  const data = (tahap: string) => r.proses.find((p) => p.tahap === tahap)?.data as Record<string, unknown> | undefined;
  return {
    alat: aksi.alat, kueri: aksi.kueri, ms: Math.round(ms), jalur: r.jalur, keyakinan: r.keyakinan ?? null, label: r.label ?? null,
    teks: r.teks.length > 1500 ? r.teks.slice(0, 1500) + "…" : r.teks,
    sumber: r.sumber.slice(0, 8).map((x) => x.label),
    skor: Number(nilai.skor.toFixed(3)), cocok: nilai.cocok, alasan: nilai.alasan,
    kurang: nilai.kurang.map((k) => ({ jenis: k.jenis, rincian: k.rincian })),
    perbaikan: lanjut.map((x) => ({ alat: x.alat, kueri: x.kueri, pikir: x.pikir })),
    samaDengan,
    proses: r.proses.filter((p) => !["Tokenisasi pesan"].includes(p.tahap)).map((p) => ({ tahap: p.tahap, hasil: p.hasil })),
    halaman: (data("Cari di internet")?.halaman as unknown[]) ?? null,
    panggilan: (data("Cari di internet")?.panggilan as unknown[]) ?? null,
    kalimat: (data("Nilai kalimat internet")?.kalimat as unknown[]) ?? null,
    kandidat: (r.proses.find((p) => p.tahap.startsWith("Cari di dokumen") || p.tahap.startsWith("Cari di Wikipedia"))?.data as { kandidat?: unknown[] } | undefined)?.kandidat ?? null,
  };
}

const ringkas = (t: string) => { const s = inti(t).split("\n").find((x) => x.trim()) ?? ""; return s.length > 150 ? s.slice(0, 147) + "…" : s; };

export async function jalankanModeAgent(mesin: MesinChat, pesanAsli: string, opsi: OpsiAgent = {}): Promise<Respons> {
  const t0 = performance.now();
  const internet = opsi.internet !== false;
  const maksLangkah = opsi.maksLangkah ?? 8, maksMs = (opsi.maksDetik ?? 20) * 1000;
  const { internet: _i, maksLangkah: _m, maksDetik: _d, ...opsiProses } = opsi;
  const ctx: KonteksAlat = { mesin, opsi: opsiProses, halaman: [] };

  // 1. Pahami maksud
  const niat = pahamiNiat(mesin, pesanAsli, opsi.topik ?? null);
  const langkah: (LangkahAgent & { detail?: unknown })[] = [];
  const t0Alat = () => performance.now();
  const percobaan: Percobaan[] = [];
  const sudah = new Set<string>();
  const antrian = rencanaAwal(niat, pesanAsli);
  const dijalankan = new Set<NamaAlat>();
  const subjekNama = niat.entitas.join(" ");
  const kumpulWajib: Aksi[] = niat.relasi && (niat.relasi.berwaktu || niat.mintaKini) && niat.entitas.length ? [
    { alat: "wiki-subjek", kueri: subjekNama, pikir: `Jawaban bisa berubah dari waktu ke waktu -> kumpulkan dulu artikel tentang ${subjekNama}.` },
    { alat: "wiki-artikel", kueri: "", pikir: `Baca artikel lengkap ${subjekNama} untuk riwayat ${niat.relasi.nama}nya.` },
    { alat: "wiki-kata-kunci", kueri: niat.kataCari.join(" "), pikir: "Cari artikel lain yang menyebut nama ini (mis. artikel lembaga tempatnya bekerja)." },
  ] : [];

  // 2–4. Loop
  while (antrian.length && percobaan.length < maksLangkah && performance.now() - t0 < maksMs) {
    const aksi = antrian.shift()!;
    if (!alatTersedia(aksi.alat, ctx, internet)) {
      // situs terblokir jaringan: catat sekali di panel cara kerja, lalu lanjut ke alat berikutnya
      const blokir = internet ? alasanDilewati(aksi.alat) : null;
      if (blokir && !sudah.has(`blokir|${aksi.alat}`)) {
        sudah.add(`blokir|${aksi.alat}`);
        langkah.push({ pikir: aksi.pikir, aksi: `${aksi.alat} "${aksi.kueri}"`, hasil: "dilewati, tidak dikirim", cek: `✗ ${blokir}`, rincian: [] });
      }
      continue;
    }
    if (aksi.alat === "wiki-artikel" && !aksi.kueri) aksi.kueri = judulArtikel(niat, ctx.halaman) ?? "";
    if (!aksi.kueri.trim()) continue;
    const kunci = `${aksi.alat}|${aksi.kueri.toLowerCase()}`;
    if (sudah.has(kunci)) continue;
    sudah.add(kunci);

    let r: Respons;
    const mulaiAlat = t0Alat();
    try { r = await jalankanAlat(aksi, niat, ctx); }
    catch (e) {
      langkah.push({ pikir: aksi.pikir, aksi: `${aksi.alat} "${aksi.kueri}"`, hasil: `gagal: ${(e as Error).message}`, cek: "✗ -> alat berikutnya", rincian: [] });
      continue;
    }
    const nilai = bandingkan(niat, r);
    percobaan.push({ aksi, r, nilai });
    const sama = percobaan.slice(0, -1).find((p) => p.r.jalur === r.jalur && p.r.teks === r.teks);
    const lanjut = nilai.cocok ? [] : perbaikan(niat, nilai.kurang, ctx, pesanAsli).filter((x) => alatTersedia(x.alat, ctx, internet) && !sudah.has(`${x.alat}|${x.kueri.toLowerCase()}`));
    langkah.push({
      pikir: aksi.pikir,
      aksi: `${aksi.alat}${aksi.kueri !== pesanAsli ? ` "${aksi.kueri}"` : ""}`,
      hasil: `jalur ${r.jalur}: ${ringkas(r.teks)}`,
      cek: nilai.cocok
        ? `cocok ✓ (skor ${nilai.skor.toFixed(2)}: ${nilai.alasan})`
        : `belum cocok ✗ (skor ${nilai.skor.toFixed(2)}${nilai.kurang.length ? `; kurang: ${nilai.kurang.map((k) => k.rincian).join("; ")}` : ""})${sama ? " — sama dengan hasil sebelumnya" : ""}${lanjut.length ? ` -> perbaiki: ${lanjut.map((x) => x.alat).join(", ")}` : ""}`,
      rincian: r.proses.filter((p) => !["Tokenisasi pesan", "Riwayat & memori"].includes(p.tahap)).map((p) => `${p.tahap}: ${p.hasil}`).slice(-6),
      detail: detailAlat(aksi, r, nilai, lanjut, performance.now() - mulaiAlat, sama ? sama.aksi.alat : null),
    });
    dijalankan.add(aksi.alat);
    if (nilai.cocok && JALUR_PASTI.has(r.jalur)) break;
    if (nilai.cocok && ["aturan", "wiki", "internet"].includes(r.jalur)) {
      // Pertanyaan yang jawabannya berubah dari waktu ke waktu ("di mana SEKARANG X bekerja?"):
      // satu kalimat yang cocok belum cukup. Kumpulkan dulu artikel subjek, artikel lengkap, dan artikel
      // terkait, baru disimpulkan dari semua fakta (reasoner.ts).
      const belum = kumpulWajib.filter((x) => !dijalankan.has(x.alat) && alatTersedia(x.alat, ctx, internet));
      if (!belum.length) break;
      antrian.unshift(...belum);
      continue;
    }
    if (r.jalur === "klarifikasi") break; // pertanyaan ambigu: lebih baik tanya balik daripada menebak
    antrian.unshift(...lanjut);
  }
  if (!percobaan.length) {
    const r = mesin.proses(pesanAsli, { ...opsiProses, mode: "otomatis" });
    percobaan.push({ aksi: { alat: "otomatis", kueri: pesanAsli, pikir: "cadangan" }, r, nilai: bandingkan(niat, r) });
  }

  // 5. Simpulkan
  const hasil = simpulkan(niat, percobaan, { halaman: ctx.halaman, pembaca: mesin.pembaca });
  const habis = percobaan.length >= maksLangkah ? `batas ${maksLangkah} langkah` : performance.now() - t0 >= maksMs ? "batas waktu" : null;
  langkah.push({
    pikir: "Bandingkan semua hasil dan simpulkan yang terbaik.",
    aksi: "simpulkan",
    hasil: hasil.bukti.length
      ? `${hasil.bukti.length} kalimat bukti dari ${[...new Set(hasil.bukti.map((b) => b.jenis))].join(", ")}${hasil.singkat ? `; jawaban singkat "${hasil.singkat}"` : ""}`
      : `jawaban dari alat ${hasil.utama.aksi.alat} (jalur ${hasil.r.jalur})`,
    cek: hasil.yakin ? `cocok ✓ skor ${hasil.utama.nilai.skor.toFixed(2)} ≥ ${AMBANG_PUAS}` : `belum ada yang cocok${habis ? ` (${habis})` : ""} -> jawaban terbaik + pertanyaan balik`,
    rincian: [
      ...[...percobaan].sort((a, b) => b.nilai.skor - a.nilai.skor).map((p, i) => `${i + 1}. ${p.aksi.alat.padEnd(15)} ${p.nilai.skor.toFixed(2)} ${p.nilai.cocok ? "✓" : "✗"} ${p.r.jalur} — ${p.nilai.alasan}`),
      ...(hasil.bukti.length ? ["Kalimat bukti terpilih (dinilai terhadap maksud Anda):", ...hasil.bukti.map((b, i) => `  [${i + 1}] skor ${b.skor.toFixed(2)} (${b.jenis}) ${b.teks.slice(0, 140)}`)] : []),
    ],
    detail: { simpulkan: true, kolam: hasil.kolam, bukti: hasil.bukti, singkat: hasil.singkat, caraSingkat: hasil.caraSingkat, yakin: hasil.yakin, utama: hasil.utama.aksi.alat },
  });

  const pahami: LangkahProses = {
    tahap: "Agent: pahami maksud", ms: 0,
    hasil: `${niat.jenis}; bentuk jawaban: ${niat.bentuk}; kata kunci: ${niat.kataKunci.join(", ") || "-"}${niat.kataSubjek.length ? ` (+ subjek: ${niat.kataSubjek.join(" ")})` : ""}${niat.entitas.length ? `; nama diri: ${niat.entitas.join(", ")}` : ""}; sumber: ${niat.domain}`,
    data: { niat: { ...niat, satuan: niat.satuan ? String(niat.satuan) : null } },
  };
  const ringkasan: LangkahProses = {
    tahap: "Agent multi-mode", ms: performance.now() - t0,
    hasil: `${percobaan.length} alat dicoba -> ${hasil.yakin ? "cocok" : "belum yakin"}: ${hasil.utama.aksi.alat} (skor ${hasil.utama.nilai.skor.toFixed(2)})`,
    data: { langkah, multiMode: true },
  };
  const r = hasil.r;
  return {
    ...r,
    proses: [pahami, ringkasan, ...r.proses],
    topik: r.topik ?? opsi.topik ?? null,
    label: `${r.label ? r.label + " · " : ""}agent: ${percobaan.length} langkah, ${hasil.yakin ? "cocok" : "belum yakin"}`,
  };
}

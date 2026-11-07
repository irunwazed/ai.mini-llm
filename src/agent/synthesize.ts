/**
 * LANGKAH 5 AGENT: SIMPULKAN dari SEMUA bukti yang terkumpul.
 *
 * Agent tidak meneruskan jawaban mode lain apa adanya. Semua hasil alat dikumpulkan menjadi satu
 * KOLAM BUKTI:
 *   - pasal dari dokumen aturan, potongan Wikipedia lokal, catatan memori (bila relevan)
 *   - SEMUA halaman internet yang terkumpul selama loop (bukan hanya kalimat terbaik satu alat)
 * Lalu setiap kalimat dinilai terhadap maksud pengguna (BM25 mini + bentuk jawaban + nama diri +
 * judul sumber + kecocokan sumber dengan jenis pertanyaan), dan agent MENYUSUN jawabannya sendiri:
 *   1. jawaban singkat sesuai bentuk yang ditanya ("3 bulan", "6 Juni 1901", "Andrea Hirata"),
 *      diambil oleh GPT pembaca (jika ada & yakin) atau pola bentuk jawaban
 *   2. kalimat bukti terbaik dari beberapa sumber, bernomor [n]
 *   3. daftar sumber + jejak alat yang dicoba
 * Jalur pasti (sapaan, maksud, hitungan, tanya balik) tetap memakai jawaban alatnya sendiri.
 * Jika tidak ada kalimat yang cukup cocok: jawaban terbaik + "belum yakin" + pertanyaan balik.
 */
import { besaran } from "../answer/question-type.js";
import { mirip } from "../answer/sentences.js";
import { pilihKalimatWeb } from "../answer/web-extract.js";
import type { Respons, SumberTampil } from "../chat-engine.js";
import { RE_TANGGAL } from "../gpt/reader-data.js";
import { baca } from "../gpt/reader.js";
import type { Pembaca } from "../rag/pipeline.js";
import type { HalamanWeb } from "../search/internet.js";
import { inti, kalimatCukup, type Penilaian } from "./compare.js";
import type { Niat } from "./intent.js";
import { penalaranFakta } from "./reasoner.js";
import type { Aksi } from "./tools.js";

export interface Percobaan { aksi: Aksi; r: Respons; nilai: Penilaian }
type JenisSumber = "aturan" | "wiki" | "internet" | "memori";
interface Dokumen { judul: string; label: string; teks: string; jenis: JenisSumber; bobot: number; dari: Percobaan | null; aturan?: boolean }
export interface KolamTampil {
  dokumen: { label: string; jenis: JenisSumber; bobot: number; panjang: number; dari: string }[];
  kandidat: { teks: string; sumber: string; jenis: JenisSumber; skor: number; cocok: string[]; lolos: boolean }[];
  penalaran?: { relasi: string; nama: string; alasan: string[]; terpilih: string; fakta: { nilai: string; waktu: string; awal: number | null; akhir: number | null; berjalan: boolean; kalimat: string; sumber: string }[] } | null;
}
export interface BuktiTerpilih { teks: string; label: string; judul: string; jenis: JenisSumber; skor: number }

const NAMA_JENIS: Record<JenisSumber, string> = { aturan: "dokumen aturan", wiki: "Wikipedia lokal", internet: "internet", memori: "catatan Anda" };
const BUKAN_NAMA = /^(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember|Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu|Presiden|Gubernur|Raja|Ratu|Pasal|Peraturan|Undang-Undang)$/;
const JALUR_PASTI = new Set(["maksud", "detail", "penalaran", "obrolan", "klarifikasi"]);

/** Kumpulkan semua sumber dari semua percobaan + semua halaman internet (tanpa duplikat). */
function kolamBukti(niat: Niat, semua: Percobaan[], halaman: HalamanWeb[]): Dokumen[] {
  const dok: Dokumen[] = [];
  const ada = new Set<string>();
  // bobot sumber: pertanyaan aturan lebih percaya dokumen aturan; pertanyaan fakta lebih percaya ensiklopedia
  const bobotJenis = (j: JenisSumber) =>
    niat.jenis === "aturan" ? { aturan: 1.3, wiki: 0.7, internet: 0.7, memori: 1 }[j]
      : niat.jenis === "fakta" ? { aturan: 0.6, wiki: 1, internet: 1, memori: 0.8 }[j]
        : niat.jenis === "pribadi" ? { aturan: 1.1, wiki: 0.7, internet: 0.7, memori: 1.3 }[j] : 1;
  for (const p of semua) {
    const j = p.r.jalur;
    if (j !== "aturan" && j !== "wiki" && j !== "memori") continue;
    for (const s of p.r.sumber) {
      if (ada.has(s.label)) continue;
      ada.add(s.label);
      dok.push({ judul: s.label.replace(/\s—\s.*$/, ""), label: s.label, teks: s.teks, jenis: j, bobot: bobotJenis(j), dari: p, aturan: j === "aturan" });
    }
  }
  for (const h of halaman) {
    const label = `${h.judul} — ${h.url} (${h.asal})`;
    if (ada.has(label)) continue;
    ada.add(label);
    dok.push({ judul: h.judul, label, teks: h.teks, jenis: "internet", bobot: bobotJenis("internet"), dari: null });
  }
  return dok;
}

/** Ambil jawaban singkat sesuai bentuk pertanyaan dari kalimat bukti teratas. */
function jawabanSingkat(niat: Niat, bukti: { teks: string; judul: string }[], pembaca: Pembaca | null): { teks: string; cara: string } | null {
  // hanya pertanyaan berjawaban pendek; "apa itu …", "bagaimana …", syarat/prosedur dijawab kalimat buktinya
  if (!["angka", "waktu", "nama", "tempat"].includes(niat.bentuk)) return null;
  const cocokBentuk = (s: string) =>
    niat.bentuk === "angka" ? /\d/.test(s) : niat.bentuk === "waktu" ? /\d|abad/.test(s)
      : niat.bentuk === "nama" || niat.bentuk === "tempat" ? /\p{Lu}/u.test(s) : true;
  const tanya = new Set([...niat.kataKunci, ...niat.kataSubjek, ...niat.entitas].map((x) => x.toLowerCase()));
  const pola = (k: string): string | null => {
    if (niat.bentuk === "angka") {
      // beberapa besaran dalam satu kalimat ("bekerja 5 tahun … cuti besar paling lama 3 bulan"):
      // pilih yang paling DEKAT dengan kata yang ditanya, apalagi jika didahului "lama/sebesar/sebanyak"
      const calon = besaran(k).filter((x) => !niat.satuan || niat.satuan.test(x.toLowerCase()));
      if (calon.length > 1) {
        const rendah = k.toLowerCase();
        const posKata = niat.kataKunci.flatMap((w) => [...rendah.matchAll(new RegExp(`\\b${w}`, "g"))].map((m) => m.index!));
        const jarak = (x: string) => {
          const i = rendah.indexOf(x.toLowerCase());
          const d = posKata.length ? Math.min(...posKata.map((p) => Math.abs(p - i))) : i;
          return d - (/(lama|lamanya|selama|sebesar|sebanyak|adalah)\s*$/.test(rendah.slice(0, i)) ? 30 : 0);
        };
        return calon.sort((a, b) => jarak(a) - jarak(b))[0];
      }
      const b = calon[0];
      if (b) return b;
      // tanpa satuan khusus: "<angka> <kata yang ditanya>" ("38 provinsi")
      const kata = niat.kataKunci.filter((w) => !/^\d+$/.test(w)).join("|");
      if (!niat.satuan && kata) { const m = k.match(new RegExp(`\\d[\\d.,]*\\s+(?:\\S+\\s+)?(?:${kata})\\b`, "i")); if (m) return m[0]; }
      if (niat.satuan) return k.match(new RegExp(String.raw`\d[\d.,]*\s?(?:${niat.satuan.source})`, "i"))?.[0] ?? null;
      return null;
    }
    if (niat.bentuk === "waktu") return k.match(RE_TANGGAL)?.[1] ?? null;
    // nama/tempat: rangkaian kata berhuruf kapital yang bukan kata pertanyaan, bukan nama bulan/hari
    for (const m of k.matchAll(/(?:\p{Lu}[\p{L}.'-]*\s?){1,4}/gu)) {
      const n = m[0].trim();
      if (m.index === 0 || n.split(/\s+/).every((w) => tanya.has(w.toLowerCase()) || BUKAN_NAMA.test(w))) continue;
      return n;
    }
    return null;
  };
  const pembacaYakin = (k: string): string | null => {
    if (!pembaca) return null;
    const h = baca(pembaca.model, pembaca.tok, niat.kueri, k, pembaca.hurufKecil);
    return h.jawaban && h.keyakinan >= pembaca.ambang && cocokBentuk(h.jawaban) ? h.jawaban : null;
  };
  // Kalimat bukti dibaca BERURUTAN dari yang paling cocok. Angka/tanggal: pola dulu (pasti), lalu pembaca.
  // Nama/tempat: GPT pembaca dulu (paham konteks "istri dari …"), lalu pola huruf kapital.
  for (const { teks: k, judul } of bukti) {
    // "siapa …?" dijawab kalimat berkata ganti ("Ia adalah istri …"): yang dimaksud = judul artikelnya
    if (niat.bentuk === "nama" && /^(Ia|Dia|Beliau)\b/.test(k) && /^\p{Lu}/u.test(judul)) return { teks: judul.replace(/\s*\(.*\)$/, ""), cara: "subjek kalimat = judul artikel" };
    const urutan = niat.bentuk === "angka" || niat.bentuk === "waktu"
      ? [[pola(k), "pola bentuk jawaban"], [pembacaYakin(k), "GPT pembaca"]] : [[pembacaYakin(k), "GPT pembaca"], [pola(k), "pola huruf kapital"]];
    for (const [t, cara] of urutan) if (t) return { teks: t, cara: cara! };
  }
  return null;
}

export function simpulkan(niat: Niat, semua: Percobaan[], opsi: { halaman: HalamanWeb[]; pembaca: Pembaca | null }):
  { r: Respons; utama: Percobaan; bukti: BuktiTerpilih[]; yakin: boolean; singkat: string | null; kolam: KolamTampil | null; caraSingkat: string | null } {
  const urut = [...semua].sort((a, b) => b.nilai.skor - a.nilai.skor);
  const utama = urut.find((p) => p.nilai.cocok) ?? urut[0];
  const jejak = semua.length > 1 ? `Jejak agent: ${semua.map((p) => `${p.aksi.alat}${p.aksi.kueri !== niat.pesan ? ` "${p.aksi.kueri}"` : ""} ${p.nilai.cocok ? "✓" : "✗"}`).join(" → ")}` : "";

  // Sapaan, pemahaman maksud, hitungan, tanya balik: jawaban alatnya sendiri sudah tepat
  if (JALUR_PASTI.has(utama.r.jalur) && (utama.nilai.cocok || niat.jenis === "obrolan" || niat.jenis === "diri"))
    return { r: utama.r, utama, bukti: [], yakin: utama.nilai.cocok, singkat: null, kolam: null, caraSingkat: null };

  // Kolam bukti -> kalimat dinilai terhadap maksud pengguna
  const dok = kolamBukti(niat, semua, opsi.halaman);
  // untuk panel cara kerja: dokumen di kolam + kalimat teratas (lolos maupun tidak)
  const kolam: KolamTampil = {
    dokumen: dok.map((d) => ({ label: d.label, jenis: d.jenis, bobot: d.bobot, panjang: d.teks.length, dari: d.dari?.aksi.alat ?? "internet" })),
    kandidat: [],
  };
  const cukup = kalimatCukup(niat);
  const kandidat = pilihKalimatWeb(niat.kueri, dok, 12, cukup).kalimat.filter(cukup)
    .map((k) => ({ ...k, skor: k.skor * dok[k.halaman].bobot }))
    .sort((a, b) => b.skor - a.skor);
  kolam.kandidat = pilihKalimatWeb(niat.kueri, dok, 12, undefined, true).kalimat
    .map((k) => ({ teks: k.teks.slice(0, 260), sumber: dok[k.halaman].label, jenis: dok[k.halaman].jenis, skor: Number((k.skor * dok[k.halaman].bobot).toFixed(2)), cocok: k.cocok, lolos: cukup(k) }))
    .sort((a, b) => b.skor - a.skor);
  const bukti: BuktiTerpilih[] = [];
  for (const k of kandidat) {
    if (bukti.length >= 3 || (bukti.length && k.skor < 0.5 * bukti[0].skor)) break;
    if (bukti.some((b) => mirip(b.teks, k.teks) > 0.5)) continue;
    const d = dok[k.halaman];
    bukti.push({ teks: k.teks, label: d.label, judul: d.judul, jenis: d.jenis, skor: k.skor });
  }

  // PENALARAN ATAS FAKTA: hubungan yang berubah dari waktu ke waktu ("di mana sekarang X bekerja?").
  // Jawaban tidak dikutip dari satu kalimat, tetapi disimpulkan dari urutan waktu SEMUA fakta di kolam.
  const nalar = niat.relasi && (niat.relasi.berwaktu || niat.mintaKini) ? penalaranFakta(niat, dok) : null;
  if (nalar) {
    kolam.penalaran = { relasi: nalar.relasi, nama: nalar.nama, alasan: nalar.alasan, terpilih: nalar.terpilih.nilai,
      fakta: nalar.fakta.map((f) => ({ nilai: f.nilai, waktu: f.waktuTeks, awal: f.awal, akhir: f.akhir, berjalan: f.berjalan, kalimat: f.kalimat.slice(0, 260), sumber: [f.label, ...f.labelLain].join(" + ") })) };
    const kronologi = [...nalar.fakta].sort((a, b) => (a.awal ?? 9999) - (b.awal ?? 9999));
    const nomor = new Map<string, number>();
    for (const f of [nalar.terpilih, ...kronologi]) for (const l of [f.label, ...f.labelLain]) if (!nomor.has(l)) nomor.set(l, nomor.size + 1);
    const cite = (f: (typeof nalar.fakta)[number]) => [f.label, ...f.labelLain].map((l) => `[${nomor.get(l)}]`).join("");
    const sumber: SumberTampil[] = [...nomor].map(([label, n]) => ({ nomor: n, label, teks: dok.find((d) => d.label === label)!.teks }));
    const ada = (j: string) => nalar.fakta.some((f) => f.jenis === j);
    const baris = [
      `${nalar.kesimpulan} ${cite(nalar.terpilih)}`, "",
      "Alasan kesimpulan:", ...nalar.alasan.map((a) => `• ${a}`), "",
      "Riwayat yang ditemukan (urut waktu):",
      ...kronologi.map((f) => `• ${f.waktuTeks || "waktu tidak disebut"}: ${f.nilai} ${cite(f)}${f === nalar.terpilih ? "  ← keadaan terkini" : ""}`),
      "", "Sumber:", ...sumber.map((s) => `  [${s.nomor}] ${s.label} (${NAMA_JENIS[(dok.find((d) => d.label === s.label)!.jenis)]})`),
      ...(ada("internet") ? ["", "Catatan: kesimpulan dari kutipan internet, bukan dasar hukum. Periksa sumbernya."] : []),
      ...(jejak ? ["", jejak] : []),
    ];
    const asal = dok.find((d) => d.label === nalar.terpilih.label)!;
    const jalurNalar = asal.jenis;
    return {
      r: {
        jalur: jalurNalar, teks: baris.join("\n"), sumber, proses: (asal.dari ?? utama).r.proses,
        topik: { pertanyaan: niat.pesan, jalur: jalurNalar === "aturan" ? "aturan" : jalurNalar === "wiki" ? "wiki" : "internet", subjek: nalar.nama, sumberLabel: asal.label, sumberTeks: asal.teks.slice(0, 6000) },
        keyakinan: nalar.terpilih.berjalan ? "tinggi" : "sedang",
        label: `disimpulkan agent dari ${nalar.fakta.length} fakta "${nalar.relasi}" (penalaran urutan waktu)`,
      },
      utama, yakin: true, singkat: nalar.terpilih.nilai, kolam, caraSingkat: "penalaran atas urutan waktu fakta",
      bukti: nalar.fakta.map((f) => ({ teks: f.kalimat, label: f.label, judul: f.judul, jenis: f.jenis as JenisSumber, skor: f === nalar.terpilih ? 1 : 0.5 })),
    };
  }

  if (!bukti.length) {
    // Tidak ada kalimat yang cukup cocok di semua sumber: jujur, jawaban terbaik + pertanyaan balik
    const r = utama.r;
    const bagian = [
      ...(r.jalur !== "tidak-ditemukan" ? ["Saya belum menemukan jawaban yang benar-benar cocok. Yang paling mendekati:\n"] : []),
      r.teks, ...(jejak ? ["", jejak] : []), "", pertanyaanBalik(niat, utama),
    ];
    return { r: { ...r, teks: bagian.join("\n"), keyakinan: "rendah" }, utama, bukti: [], yakin: false, singkat: null, kolam, caraSingkat: null };
  }

  // Susun jawaban agent sendiri
  const singkat = jawabanSingkat(niat, bukti, opsi.pembaca);
  const nomor = new Map<string, number>();
  for (const b of bukti) if (!nomor.has(b.label)) nomor.set(b.label, nomor.size + 1);
  const sumber: SumberTampil[] = [...nomor].map(([label, n]) => ({ nomor: n, label, teks: dok.find((d) => d.label === label)!.teks }));
  const jenisDipakai = [...new Set(bukti.map((b) => b.jenis))];
  const baris: string[] = [];
  if (singkat) baris.push(`${singkat.teks.charAt(0).toUpperCase()}${singkat.teks.slice(1)}${/[.!?]$/.test(singkat.teks) ? "" : "."}`);
  baris.push(`${bukti[0].teks} [${nomor.get(bukti[0].label)}]`);
  if (bukti.length > 1) { baris.push("", "Informasi lain yang cocok:"); for (const b of bukti.slice(1)) baris.push(`• ${b.teks} [${nomor.get(b.label)}]`); }
  baris.push("", "Sumber:", ...sumber.map((s) => `  [${s.nomor}] ${s.label} (${NAMA_JENIS[bukti.find((b) => b.label === s.label)!.jenis]})`));
  if (jenisDipakai.includes("internet")) baris.push("", "Catatan: kutipan dari internet bukan dasar hukum. Periksa sumbernya.");
  if (jejak) baris.push("", jejak);

  const jalur = bukti[0].jenis;
  const asal = dok.find((d) => d.label === bukti[0].label)!;
  const topik: Respons["topik"] = asal.dari?.r.topik ?? (jalur === "internet"
    ? { pertanyaan: niat.pesan, jalur: "internet", subjek: asal.judul.replace(/\s*\(.*\)$/, ""), sumberLabel: asal.label, sumberTeks: asal.teks.slice(0, 6000) }
    : utama.r.topik);
  // bukti sudah lolos syarat kata kunci; untuk "berapa/kapan/siapa/di mana" jawaban singkatnya juga harus ditemukan
  const yakin = utama.nilai.cocok || !!singkat || !["angka", "waktu", "nama", "tempat"].includes(niat.bentuk);
  return {
    r: {
      jalur, teks: baris.join("\n"), sumber, proses: (asal.dari ?? utama).r.proses, topik,
      keyakinan: utama.nilai.cocok ? "tinggi" : "sedang",
      label: `disimpulkan agent dari ${bukti.length} kalimat bukti (${jenisDipakai.map((j) => NAMA_JENIS[j]).join(", ")})${singkat ? `; jawaban singkat: ${singkat.cara}` : ""}`,
    },
    utama, bukti, yakin, singkat: singkat?.teks ?? null, kolam, caraSingkat: singkat?.cara ?? null,
  };
}

function pertanyaanBalik(niat: Niat, terbaik: Percobaan): string {
  const k = terbaik.nilai.kurang;
  const entitas = k.find((x) => x.jenis === "entitas");
  if (entitas) return `Yang Anda maksud ${entitas.kata!.join(", ")} yang mana? Tambahkan keterangan (misalnya bidang, tahun, atau tempat) agar saya bisa mencari lebih tepat.`;
  const kata = k.find((x) => x.jenis === "kata");
  if (kata) return `Saya belum menemukan keterangan tentang "${kata.kata!.join(", ")}". Bisa dijelaskan lebih rinci apa yang ingin Anda ketahui?`;
  if (k.some((x) => x.jenis === "sumber") || niat.domain === "tidak-tahu") return "Apakah yang Anda tanyakan aturan pemerintah (dokumen), atau pengetahuan umum? Atau coba sebutkan kata kunci yang lebih khusus.";
  return "Bisa dijelaskan dengan kata lain, atau sebutkan kata kunci yang lebih khusus?";
}

export { inti };

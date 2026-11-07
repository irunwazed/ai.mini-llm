/**
 * SILOGISME (logika kuantor) dengan himpunan:
 *   "Semua kucing adalah hewan"            -> kucing ⊆ hewan
 *   "Beberapa hewan bisa berenang"         -> hewan ∩ perenang ≠ ∅   (TIDAK berarti semua)
 *   "Tidak ada ikan yang bisa terbang"     -> ikan ∩ penerbang = ∅
 * Pertanyaan "apakah semua kucing bisa berenang?" valid HANYA jika ada rantai "semua":
 *   kucing ⊆ … ⊆ perenang.
 * Jika tidak ada rantai seperti itu, kesimpulan tidak bisa dipastikan, dan dibuatkan
 * CONTOH PENYANGKAL: dunia kecil yang memenuhi semua premis tetapi kesimpulannya salah.
 */
import type { Alat } from "./types.js";

type Relasi = { jenis: "semua" | "beberapa" | "tidak"; a: string; b: string };

const bersih = (s: string) => s.toLowerCase().replace(/[?.!,]/g, "").replace(/\b(yang|adalah|merupakan|itu|juga)\b/g, " ").replace(/\s+/g, " ").trim();

function premis(t: string): Relasi[] {
  const hasil: Relasi[] = [];
  for (const k of t.split(/[.,;]|\bdan\b/i)) {
    const s = k.trim().toLowerCase();
    let m = s.match(/^(?:jika\s+)?semua\s+(.+?)\s+(?:adalah|merupakan|bisa|dapat|suka|punya|memiliki)\s+(.+)$/);
    if (m) { hasil.push({ jenis: "semua", a: bersih(m[1]), b: bersih(s.includes(" bisa ") || s.includes(" dapat ") ? `bisa ${m[2]}` : m[2]) }); continue; }
    m = s.match(/^(?:jika\s+)?(?:beberapa|sebagian)\s+(.+?)\s+(?:adalah|merupakan|bisa|dapat|suka|punya|memiliki)\s+(.+)$/);
    if (m) { hasil.push({ jenis: "beberapa", a: bersih(m[1]), b: bersih(s.includes(" bisa ") || s.includes(" dapat ") ? `bisa ${m[2]}` : m[2]) }); continue; }
    m = s.match(/^(?:jika\s+)?tidak\s+ada\s+(.+?)\s+yang\s+(?:adalah\s+|bisa\s+|dapat\s+)?(.+)$/);
    if (m) hasil.push({ jenis: "tidak", a: bersih(m[1]), b: bersih(m[2]) });
  }
  return hasil;
}

export const silogisme: Alat = {
  nama: "silogisme (himpunan)",
  pikir: "Soal memakai kata 'semua/beberapa/tidak ada', jadi ubah setiap kalimat menjadi hubungan himpunan.",
  cocok: (t) => /\b(semua|beberapa|sebagian|tidak ada)\b/i.test(t) && /apakah\s+(semua|beberapa|sebagian|ada)/i.test(t),
  coba: (t) => {
    const [bagianPremis, bagianTanya] = t.split(/apakah/i);
    const rel = premis(bagianPremis);
    const q = bagianTanya?.toLowerCase().match(/^\s*(semua|beberapa|sebagian)\s+(.+?)\s+(?:adalah|merupakan|bisa|dapat)\s+(.+?)\??$/);
    if (!rel.length || !q) return null;
    const kuantor = q[1], A = bersih(q[2]), B = bersih(bagianTanya.includes(" bisa ") || bagianTanya.includes(" dapat ") ? `bisa ${q[3]}` : q[3]);
    const penjelasan = rel.map((r) => r.jenis === "semua" ? `${r.a} ⊆ ${r.b}  (semua ${r.a} termasuk ${r.b})` : r.jenis === "beberapa" ? `${r.a} ∩ ${r.b} ≠ ∅  (ada sebagian ${r.a} yang ${r.b}, tidak harus semua)` : `${r.a} ∩ ${r.b} = ∅  (tidak ada yang sama)`);
    // cari rantai "semua" dari A ke B
    const semua = rel.filter((r) => r.jenis === "semua");
    const kunjung = new Set([A]), antre = [A];
    while (antre.length) { const x = antre.shift()!; for (const r of semua) if (r.a === x && !kunjung.has(r.b)) { kunjung.add(r.b); antre.push(r.b); } }
    if (kunjung.has(B))
      return { jawaban: `Ya, ${kuantor} ${A} ${B}.`, penjelasan: [...penjelasan, `Ada rantai "semua" dari ${A} ke ${B}, jadi kesimpulan pasti benar.`], cek: { lolos: true, teks: "rantai subset ditemukan" } };
    const jembatan = rel.find((r) => r.jenis === "beberapa" && kunjung.has(r.a));
    return {
      jawaban: "Tidak bisa disimpulkan.",
      penjelasan: [
        ...penjelasan,
        `Tidak ada rantai "semua" dari ${A} ke ${B}.`,
        jembatan
          ? `Contoh penyangkal: bisa saja ${jembatan.a} yang ${jembatan.b} hanyalah ${jembatan.a} lain (bukan ${A}); semua premis tetap benar, tetapi tidak ada ${A} yang ${B}.`
          : `Premis tidak menghubungkan ${A} dengan ${B} sama sekali.`,
      ],
      cek: { lolos: true, teks: "contoh penyangkal memenuhi semua premis" },
    };
  },
};

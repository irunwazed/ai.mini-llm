/**
 * BANK TEKA-TEKI (jebakan, logika lanjut, paradoks) beserta penjelasannya.
 *
 * Ini bukan "berpikir": jawabannya sudah disiapkan, sama seperti model besar yang
 * menjawab teka-teki terkenal karena pernah melihatnya. Agar tidak sekadar hafalan,
 * teka-teki yang memuat angka/nama DIBUAT BERPARAMETER: angka dan nama dibaca dari soal,
 * sehingga variasinya (6 apel, 15 ikan, posisi ketiga, total Rp2.000) tetap terjawab benar.
 * Setiap entri dikenali dengan pola kata kunci yang semuanya harus ada.
 */
import { angkaDigit, kataKeAngka, rupiah } from "./numbers.js";
import type { Alat, HasilAlat } from "./types.js";

interface TekaTeki { id: string; pola: RegExp[]; jawab: (t: string) => HasilAlat | null }

const ORDINAL = ["", "pertama", "kedua", "ketiga", "keempat", "kelima", "keenam", "ketujuh", "kedelapan", "kesembilan", "kesepuluh"];
const ordinalKeAngka = (s: string) => { const i = ORDINAL.indexOf(s.toLowerCase()); return i > 0 ? i : Number(s); };

const BANK: TekaTeki[] = [
  {
    id: "mengambil-apel",
    pola: [/\d+\s+\w+/, /(mengambil|ambil)\s+\d+/i, /berapa\s+.*(kamu|anda)\s+(miliki|punya|pegang)/i],
    jawab: (t) => {
      const a = angkaDigit(t); const ambil = t.match(/(?:mengambil|ambil)\s+(\d+)\s+(\w+)/i);
      if (!ambil || a.length < 2) return null;
      return { jawaban: `${ambil[1]} ${ambil[2]}`, penjelasan: [`Yang ditanya: berapa yang KAMU miliki, bukan berapa yang tersisa.`, `Kamu mengambil ${ambil[1]}, jadi yang kamu miliki ${ambil[1]}. (Yang tersisa di tempat semula: ${a[0].nilai - Number(ambil[1])}.)`] };
    },
  },
  {
    id: "ikan-mati",
    pola: [/\d+\s+ikan/i, /\d+\s+(ikan\s+)?mati/i, /(tersisa|sisa|ada).*akuarium/i],
    jawab: (t) => {
      const n = angkaDigit(t)[0]?.nilai; if (n === undefined) return null;
      return { jawaban: `${n} ikan`, penjelasan: ["Ikan yang mati tetap berada di dalam akuarium.", `Jadi jumlah ikan di akuarium tetap ${n}.`] };
    },
  },
  {
    id: "menyalip",
    pola: [/menyalip/i, /(posisi|urutan|peringkat)\s+(\d+|ke-?\d+|pertama|kedua|ketiga|keempat|kelima|terakhir)/i],
    jawab: (t) => {
      if (/terakhir/i.test(t)) return { jawaban: "Tidak mungkin", penjelasan: ["Tidak ada yang bisa menyalip orang di posisi terakhir dari belakangnya, karena tidak ada orang di belakang posisi terakhir."] };
      const m = t.match(/(?:posisi|urutan|peringkat)\s+(?:ke-?)?(\d+|pertama|kedua|ketiga|keempat|kelima)/i)!;
      const k = ordinalKeAngka(m[1]);
      return { jawaban: `Posisi ${ORDINAL[k] ?? k}`, penjelasan: [`Saat menyalip orang di posisi ${k}, kamu menggantikan posisinya.`, `Orang itu turun ke posisi ${k + 1}, dan kamu sekarang di posisi ${k} (bukan ${k - 1}).`] };
    },
  },
  {
    id: "anak-kelima",
    pola: [/ayah\s+(\w+)\s+(memiliki|punya)\s+\d+\s+anak/i, /siapa\s+anak/i],
    jawab: (t) => {
      const m = t.match(/ayah\s+(\w+)\s+(?:memiliki|punya)\s+(\d+)\s+anak/i)!;
      return { jawaban: m[1], penjelasan: [`Kalimatnya "Ayah ${m[1]} memiliki ${m[2]} anak", jadi ${m[1]} sendiri adalah salah satu anaknya.`, `Nama-nama yang disebut hanya ${Number(m[2]) - 1} anak lain; anak yang belum disebut adalah ${m[1]}.`] };
    },
  },
  {
    id: "everest",
    pola: [/sebelum/i, /(everest)/i, /(ditemukan|diketahui)/i, /(tertinggi|paling tinggi)/i],
    jawab: () => ({ jawaban: "Gunung Everest", penjelasan: ["Gunung Everest sudah menjadi gunung tertinggi sebelum manusia 'menemukan' atau mengukurnya.", "Ditemukan atau belum, tingginya tidak berubah."] }),
  },
  {
    id: "kereta-listrik",
    pola: [/kereta\s+listrik/i, /asap/i],
    jawab: () => ({ jawaban: "Kereta listrik tidak mengeluarkan asap", penjelasan: ["Kereta listrik digerakkan listrik, bukan pembakaran, jadi tidak ada asap.", "Informasi arah angin hanya pengalih perhatian."] }),
  },
  {
    id: "tiga-sakelar",
    pola: [/(3|tiga)\s+sakelar/i, /lampu/i, /((satu|1)\s+kali|sekali)/i],
    jawab: () => ({
      jawaban: "Gunakan panas lampu sebagai petunjuk",
      penjelasan: [
        "1. Nyalakan sakelar A, tunggu beberapa menit, lalu matikan.",
        "2. Nyalakan sakelar B, lalu masuk ke ruangan.",
        "3. Lampu menyala → sakelar B. Lampu mati tetapi hangat → sakelar A. Lampu mati dan dingin → sakelar C.",
        "Kuncinya: lampu punya dua keadaan yang bisa diamati (menyala dan panas), sehingga satu kunjungan cukup membedakan 3 kemungkinan.",
      ],
    }),
  },
  {
    id: "foto-keluarga",
    pola: [/foto/i, /tidak\s+punya\s+saudara/i, /anak\s+ayah\s+saya/i],
    jawab: () => ({
      jawaban: "Anak dari pria itu (anaknya sendiri)",
      penjelasan: [
        "\"Anak ayah saya\": karena pria itu tidak punya saudara, anak ayahnya adalah dirinya sendiri.",
        "Jadi kalimatnya menjadi: \"ayah orang di foto adalah SAYA\".",
        "Maka orang di foto adalah anak pria itu.",
      ],
    }),
  },
  {
    id: "delapan-bola",
    pola: [/(8|delapan)\s+bola/i, /(lebih\s+berat|berat)/i, /(2|dua)\s+kali\s+penimbangan|menimbang\s+(2|dua)\s+kali/i],
    jawab: () => ({
      jawaban: "Bagi menjadi 3–3–2",
      penjelasan: [
        "Penimbangan 1: timbang 3 bola vs 3 bola (2 bola disisihkan).",
        "• Jika seimbang → bola berat ada di 2 sisanya. Penimbangan 2: timbang 1 vs 1, yang turun adalah bola berat.",
        "• Jika tidak seimbang → ambil 3 bola di sisi yang turun. Penimbangan 2: timbang 1 vs 1 dari ketiganya. Jika seimbang, bola ketiga yang berat; jika tidak, yang turun.",
        "Alasan matematis: setiap penimbangan punya 3 hasil (kiri, kanan, seimbang), jadi 2 penimbangan membedakan 3 × 3 = 9 ≥ 8 kemungkinan.",
      ],
    }),
  },
  {
    id: "dua-penjaga",
    pola: [/(dua|2)\s+pintu/i, /(jujur)/i, /(bohong|berbohong)/i],
    jawab: () => ({
      jawaban: "Tanyakan: \"Pintu mana yang akan ditunjuk penjaga yang lain sebagai pintu keselamatan?\" lalu pilih pintu yang SEBALIKNYA.",
      penjelasan: [
        "Jika bertanya ke penjaga jujur: ia jujur melaporkan jawaban si pembohong, yang menunjuk pintu bahaya.",
        "Jika bertanya ke pembohong: ia berbohong tentang jawaban si jujur, jadi menunjuk pintu bahaya juga.",
        "Kedua kasus selalu menunjuk pintu bahaya, sehingga pintu yang lain pasti pintu keselamatan.",
      ],
    }),
  },
  {
    id: "pernyataan-salah",
    pola: [/semua\s+pernyataan\s+(berikut|ini)\s+salah/i],
    jawab: () => ({
      jawaban: "Terjadi kontradiksi (paradoks)",
      penjelasan: [
        "Andaikan pernyataan itu benar. Jika ia termasuk 'semua pernyataan' yang dibicarakan, maka ia sendiri harus salah.",
        "Benar sekaligus salah tidak mungkin, jadi asumsi 'pernyataan itu benar' tidak bisa dipertahankan.",
        "Ini mirip paradoks pembohong: kalimat yang membicarakan kebenaran dirinya sendiri.",
      ],
    }),
  },
  {
    id: "satu-kata-tidak",
    pola: [/(satu|1)\s+kata/i, /jawaban(mu|nya)?.*adalah\s+"?tidak"?/i],
    jawab: () => ({
      jawaban: "Tidak ada jawaban satu kata yang konsisten (paradoks)",
      penjelasan: [
        "Jika menjawab \"tidak\": maka jawabannya adalah \"tidak\", sehingga seharusnya \"ya\". Kontradiksi.",
        "Jika menjawab \"ya\": maka jawabannya \"ya\", bukan \"tidak\", sehingga seharusnya \"tidak\". Kontradiksi.",
      ],
    }),
  },
  {
    id: "semua-jawaban-salah",
    pola: [/semua\s+jawaban(mu|nya)?\s+salah/i, /pasti\s+benar/i],
    jawab: () => ({
      jawaban: "Tidak, pernyataan itu tidak pasti benar.",
      penjelasan: [
        "Cukup ada SATU jawaban yang benar untuk membuat pernyataan \"semua jawabanmu salah\" menjadi salah.",
        "Jika jawaban ini (\"tidak pasti benar\") benar, maka pernyataan itu sudah terbukti salah.",
        "Jika jawaban ini salah, pernyataan itu memang bisa benar. Jadi kebenarannya tidak pasti.",
      ],
    }),
  },
  {
    id: "nol-koma-sembilan",
    pola: [/0[,.]9{3,}|0[,.]999/i, /(sama dengan|=)\s*1/i],
    jawab: () => ({
      jawaban: "0,999… = 1",
      penjelasan: [
        "Cara 1: 1/3 = 0,333…  →  dikali 3: 3/3 = 0,999…  →  padahal 3/3 = 1.",
        "Cara 2: misalkan x = 0,999…  →  10x = 9,999…  →  10x − x = 9,999… − 0,999… = 9  →  9x = 9  →  x = 1.",
        "Cara 3: jika 0,999… < 1, pasti ada angka di antara keduanya. Tidak ada angka seperti itu, jadi keduanya sama.",
      ],
    }),
  },
  {
    id: "jumlah-tanpa-angka",
    pola: [/(\w+|\d+)\s+angka/i, /(dijumlahkan|jumlah(nya)?)/i, /(menghasilkan|hasilnya|sama dengan)\s+\d+/i, /tanpa/i],
    jawab: (t) => {
      const target = Number(t.match(/(?:menghasilkan|hasilnya|sama dengan)\s+(\d+)/i)![1]);
      const larang = Number(t.match(/tanpa\s+(?:menggunakan\s+)?(?:angka\s+)?(\d+)/i)?.[1] ?? NaN);
      const banyak = Number(kataKeAngka((t.match(/(\w+)\s+contoh\s+angka|(\w+)\s+angka/i)?.slice(1).find(Boolean) ?? "3")).match(/\d+/)?.[0] ?? 3);
      // cari bilangan bulat positif berbeda, tidak memuat digit/angka terlarang, dari yang terbesar
      const ok = (x: number) => x > 0 && x !== larang && !(Number.isFinite(larang) && String(x).includes(String(larang)));
      const pilih: number[] = [];
      let sisa = target;
      for (let k = banyak; k > 1; k--) {
        let x = Math.floor(sisa / k) + (k - 1);
        // pada langkah terakhir (k = 2), sisa − x menjadi angka terakhir: harus sah dan berbeda
        const terakhirKembar = (y: number) => k === 2 && (sisa - y === y || pilih.includes(sisa - y));
        while (x > 0 && (!ok(x) || pilih.includes(x) || !ok(sisa - x) || terakhirKembar(x))) x--;
        if (x <= 0) return null;
        pilih.push(x); sisa -= x;
      }
      if (!ok(sisa) || pilih.includes(sisa)) return null;
      pilih.push(sisa);
      return {
        jawaban: pilih.join(" + ") + ` = ${target}`,
        penjelasan: [`Dibutuhkan ${banyak} angka berjumlah ${target}${Number.isFinite(larang) ? ` tanpa memakai ${larang}` : ""}.`, `Contoh: ${pilih.join(" + ")} = ${pilih.reduce((a, b) => a + b, 0)}.`, "Masih banyak contoh lain; syaratnya hanya jumlah dan angka yang dilarang."],
        cek: { lolos: pilih.reduce((a, b) => a + b, 0) === target && pilih.every(ok), teks: `${pilih.join(" + ")} = ${pilih.reduce((a, b) => a + b, 0)} ✓, tidak ada ${larang}` },
      };
    },
  },
  {
    id: "dua-koin",
    pola: [/(dua|2)\s+koin/i, /total|jumlah/i, /bukan\s+koin/i],
    jawab: (t) => {
      const angka = angkaDigit(t).map((a) => a.nilai);
      const total = angka.find((x) => x >= 200) ?? angka[0];
      const bukan = Number(t.match(/bukan\s+koin\s+(?:rp\s?)?([\d.]+)/i)?.[1].replace(/\./g, "") ?? NaN);
      const KOIN = [100, 200, 500, 1000];
      for (const a of KOIN) for (const b of KOIN) if (a + b === total && a >= b && (a === bukan || b === bukan) && a !== b) {
        const lain = a === bukan ? b : a;
        return {
          jawaban: `${rupiah(a)} dan ${rupiah(b)}`,
          penjelasan: [`Pasangan koin yang totalnya ${rupiah(total)}: ${rupiah(a)} + ${rupiah(b)}.`, `"Salah satunya bukan koin ${rupiah(bukan)}" benar: koin ${rupiah(lain)} memang bukan ${rupiah(bukan)}. Yang lainnya boleh saja ${rupiah(bukan)}.`],
          cek: { lolos: a + b === total, teks: `${rupiah(a)} + ${rupiah(b)} = ${rupiah(total)} ✓` },
        };
      }
      return null;
    },
  },
];

export const tekaTeki: Alat = {
  nama: "bank teka-teki (dikenali dari pola, angka/nama dibaca dari soal)",
  pikir: "Kalimatnya cocok dengan pola teka-teki jebakan yang dikenal, jadi pakai penjelasan yang sudah disiapkan.",
  cocok: (t) => BANK.some((b) => b.pola.every((p) => p.test(t))),
  coba: (t) => {
    for (const b of BANK) if (b.pola.every((p) => p.test(t))) { const r = b.jawab(t); if (r) return { ...r, catatan: `teka-teki "${b.id}"` }; }
    return null;
  },
};

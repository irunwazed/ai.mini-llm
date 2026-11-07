/**
 * HARI DALAM MINGGU: "Jika hari ini Rabu, 100 hari lagi hari apa?"
 *   Hari berulang setiap 7 hari, jadi yang penting hanya SISA bagi 7 (modulo):
 *   100 = 14 × 7 + 2  ->  maju 2 hari dari Rabu = Jumat
 *   indeks_baru = (indeks_awal + n) mod 7
 */
import { angkaDigit } from "./numbers.js";
import type { Alat } from "./types.js";

const HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"];
const RE_HARI = /\b(senin|selasa|rabu|kamis|jumat|jum'at|sabtu|minggu|ahad)\b/i;

export const kalender: Alat = {
  nama: "kalender (modulo 7)",
  pikir: "Soal menyebut nama hari dan jumlah hari, jadi cukup hitung sisa bagi 7.",
  cocok: (t) => RE_HARI.test(t) && /\d+\s*hari\s*(lagi|kemudian|ke depan|yang lalu|sebelumnya|sebelum)/i.test(t),
  coba: (t) => {
    const h = t.match(RE_HARI);
    const n = t.match(/(\d+)\s*hari\s*(lagi|kemudian|ke depan|yang lalu|sebelumnya|sebelum)/i);
    if (!h || !n) return null;
    const nama = h[1].toLowerCase().replace("jum'at", "jumat").replace("ahad", "minggu");
    const awal = HARI.findIndex((x) => x.toLowerCase() === nama);
    const jumlah = Number(n[1]), mundur = /lalu|sebelum/i.test(n[2]);
    const sisa = jumlah % 7, geser = mundur ? -sisa : sisa;
    const akhir = (((awal + geser) % 7) + 7) % 7;
    const hasilBalik = (((akhir - geser) % 7) + 7) % 7;
    return {
      jawaban: HARI[akhir],
      penjelasan: [
        `${HARI[awal]} = hari ke-${awal + 1} dalam minggu (Senin = 1).`,
        `${jumlah} hari = ${Math.floor(jumlah / 7)} minggu penuh + ${sisa} hari  (${jumlah} mod 7 = ${sisa}).`,
        `Minggu penuh tidak mengubah nama hari, jadi cukup ${mundur ? "mundur" : "maju"} ${sisa} hari dari ${HARI[awal]}: ${HARI[akhir]}.`,
      ],
      cek: { lolos: hasilBalik === awal, teks: `${mundur ? "maju" : "mundur"} ${sisa} hari dari ${HARI[akhir]} kembali ke ${HARI[hasilBalik]} ✓` },
    };
  },
};

export const _uji = { angkaDigit };

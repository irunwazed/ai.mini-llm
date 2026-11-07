/** Bentuk hasil yang sama untuk semua alat penalaran. */
export interface HasilAlat {
  jawaban: string;          // jawaban singkat, mis. "Jumat"
  penjelasan: string[];     // langkah hitung / alasan, satu baris per langkah
  cek?: { lolos: boolean; teks: string }; // verifikasi: jawaban dimasukkan kembali ke soal
  catatan?: string;         // mis. peringatan ambiguitas
}

export interface Alat {
  nama: string;             // nama pendek, tampil di "Aksi"
  pikir: string;            // alasan memilih alat ini, tampil di "Pikir"
  cocok: (teks: string) => boolean;   // pengenal cepat (pola kalimat)
  coba: (teks: string) => HasilAlat | null;
}

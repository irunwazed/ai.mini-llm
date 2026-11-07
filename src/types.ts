/** Hasil membaca satu file. PDF punya banyak halaman; TXT/MD/DOCX dianggap satu halaman. */
export interface Dokumen {
  file: string;          // path relatif terhadap folder dokumen, pakai "/"
  jenis: "pdf" | "docx" | "txt" | "md";
  adaHalaman: boolean;   // true untuk PDF
  halaman: string[];     // teks per halaman
}

/** Satu unit pencarian: biasanya satu pasal. */
export interface Potongan {
  id: number;
  file: string;
  judul: string;         // judul peraturan (baris awal dokumen)
  bab: string;           // mis. "BAB II CUTI TAHUNAN"
  pasal: string;         // mis. "Pasal 5" atau "Pembukaan" atau "Bagian 3"
  halaman: number | null;// halaman awal (PDF)
  baris: number;         // nomor baris awal dalam dokumen (1-based)
  teks: string;
}

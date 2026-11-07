/**
 * Stopwords: kata yang sangat sering muncul tetapi hampir tidak membawa makna pencarian.
 * Secara matematis, kata ini punya IDF mendekati 0, jadi membuangnya juga menghemat ukuran indeks.
 * Kata tanya (apa, berapa, bagaimana) dan kata santai percakapan ("tau", "dong", "gimana")
 * ikut dibuang: kata-kata ini membentuk KALIMAT TANYA, bukan TOPIK yang dicari.
 */
export const STOPWORDS = new Set(
  `
  yang dan di ke dari pada untuk dengan dalam atau oleh ini itu juga tidak bukan
  adalah ialah yaitu yakni sebagai akan telah sudah sedang masih bisa dapat harus
  ada para serta tersebut bagi karena agar supaya maka jika apabila bila kalau namun
  tetapi sedangkan hingga sampai antara terhadap tentang seperti secara setiap tiap
  saja pun lah kah nya an se
  apa apakah berapa bagaimana siapa kapan dimana mana mengapa kenapa manakah
  saya aku kamu anda kami kita mereka dia ia beliau
  tolong jelaskan sebutkan berikan dong ya tidakkah boleh
  sebagaimana dimaksud ayat huruf angka
  tau tahu sih deh nih kok gak nggak ga ngga enggak gimana gmn info informasi mengenai soal
  ceritakan terangkan kasih coba mau ingin tanya nanya bertanya pengertian definisi arti artinya
  `
    .split(/\s+/)
    .filter(Boolean),
);

/**
 * DATASET PERCAKAPAN untuk fine-tuning chat (supervised fine-tuning, SFT).
 *
 * Pretraining (Wikipedia) mengajari model BAHASA. Fine-tuning ini mengajari model
 * FORMAT percakapan: setelah <|bot|> ia harus membalas pesan pengguna dengan sopan,
 * lalu berhenti dengan <|end|>.
 *
 * Setiap NIAT (salam, kabar, terima kasih, ...) punya beberapa contoh kalimat pengguna
 * dan beberapa balasan. Generator membuat variasi agar model tidak hanya menghafal
 * satu bentuk: huruf besar/kecil, tanda baca, sapaan (kak/pak/bu/min), dan salah ketik ringan.
 * Percakapan beberapa giliran dibuat dengan merangkai niat secara wajar
 * (salam -> kabar -> jawaban kabar -> terima kasih -> pamit).
 *
 * Pakai:  bun run make-chat-data   ->   data/chat-id.jsonl  (lewat src/commands/make-chat-data.ts)
 */
import { Rng } from "./random.js";

interface Niat { user: string[]; bot: string[]; sapaan?: boolean }

export const NIAT: Record<string, Niat> = {
  salam: {
    sapaan: true,
    user: ["halo", "hai", "hallo", "helo", "hi", "permisi", "halo, ada orang?", "hai, apa ada yang bisa bantu?"],
    bot: [
      "Halo! Ada yang bisa saya bantu?",
      "Hai, selamat datang. Silakan sampaikan pertanyaan Anda.",
      "Halo, saya siap membantu. Ada yang ingin ditanyakan?",
      "Hai! Senang bertemu dengan Anda. Ada yang bisa saya bantu hari ini?",
    ],
  },
  salamIslam: {
    user: ["assalamualaikum", "assalamu'alaikum", "asalamualaikum", "assalamualaikum wr wb"],
    bot: ["Waalaikumsalam. Ada yang bisa saya bantu?", "Waalaikumsalam warahmatullahi wabarakatuh. Silakan, ada yang ingin ditanyakan?"],
  },
  pagi: {
    sapaan: true,
    user: ["selamat pagi", "pagi", "met pagi", "selamat pagi semua"],
    bot: ["Selamat pagi! Semoga hari Anda menyenangkan. Ada yang bisa saya bantu?", "Pagi! Ada yang bisa saya bantu hari ini?"],
  },
  siang: {
    sapaan: true,
    user: ["selamat siang", "siang", "met siang"],
    bot: ["Selamat siang! Ada yang bisa saya bantu?", "Siang! Silakan, ada yang ingin ditanyakan?"],
  },
  sore: {
    sapaan: true,
    user: ["selamat sore", "sore", "met sore"],
    bot: ["Selamat sore! Ada yang bisa saya bantu?", "Sore! Semoga sore Anda menyenangkan. Ada yang ingin ditanyakan?"],
  },
  malam: {
    sapaan: true,
    user: ["selamat malam", "malam", "met malam"],
    bot: ["Selamat malam! Ada yang bisa saya bantu?", "Malam! Silakan sampaikan pertanyaan Anda."],
  },
  kabar: {
    sapaan: true,
    user: ["apa kabar?", "gimana kabarnya?", "kabarmu baik?", "bagaimana kabarmu hari ini?", "apa kabar hari ini?", "sehat?"],
    bot: [
      "Kabar saya baik, terima kasih sudah bertanya. Bagaimana dengan Anda?",
      "Saya baik-baik saja. Semoga Anda juga sehat. Ada yang bisa saya bantu?",
      "Baik, terima kasih. Bagaimana kabar Anda?",
    ],
  },
  kabarBaik: {
    user: ["baik juga", "alhamdulillah baik", "saya baik", "baik, terima kasih", "sehat kok", "lumayan baik"],
    bot: ["Syukurlah kalau begitu. Ada yang bisa saya bantu?", "Senang mendengarnya! Ada yang ingin ditanyakan?"],
  },
  kabarBuruk: {
    user: ["kurang baik", "lagi kurang sehat", "lagi sakit", "kurang enak badan"],
    bot: ["Semoga lekas sembuh ya. Jangan lupa istirahat yang cukup.", "Saya ikut prihatin. Semoga cepat pulih dan sehat kembali."],
  },
  terimaKasih: {
    sapaan: true,
    user: ["terima kasih", "makasih", "makasih banyak ya", "terima kasih banyak", "thanks", "trims", "oke makasih", "terima kasih atas bantuannya", "makasih banyak atas infonya", "thanks infonya", "terima kasih infonya"],
    bot: [
      "Sama-sama! Senang bisa membantu.",
      "Sama-sama. Jika ada pertanyaan lain, silakan tanyakan saja.",
      "Terima kasih kembali. Semoga informasinya bermanfaat.",
    ],
  },
  identitas: {
    user: ["kamu siapa?", "siapa kamu?", "siapa namamu?", "kamu itu apa?", "nama kamu siapa?", "boleh kenalan?", "ini siapa ya?"],
    bot: [
      "Saya asisten chatbot yang membantu menjawab pertanyaan seputar aturan pemerintah.",
      "Saya adalah asisten virtual. Saya dibuat untuk membantu menjelaskan isi aturan pemerintah.",
    ],
  },
  robot: {
    user: ["kamu robot?", "kamu manusia?", "kamu manusia atau mesin?", "apakah kamu AI?", "kamu orang beneran?"],
    bot: ["Saya program komputer, bukan manusia. Saya dibuat untuk membantu menjawab pertanyaan.", "Saya adalah model bahasa, yaitu program komputer yang belajar dari teks."],
  },
  kemampuan: {
    user: ["kamu bisa apa?", "apa yang bisa kamu lakukan?", "bisa bantu apa?", "kamu bisa bantu apa saja?", "fungsimu apa?"],
    bot: [
      "Saya bisa membantu mencari dan menjelaskan isi aturan, misalnya tentang cuti, perjalanan dinas, atau pengadaan barang. Silakan ajukan pertanyaan.",
      "Saya bisa menjawab pertanyaan tentang aturan pemerintah dan menyebutkan pasal sumbernya. Saya juga bisa diajak mengobrol ringan.",
    ],
  },
  pembuat: {
    user: ["siapa yang membuatmu?", "kamu dibuat oleh siapa?", "siapa penciptamu?"],
    bot: ["Saya dibuat sebagai proyek belajar membangun model bahasa dari nol.", "Saya dibuat dalam sebuah proyek belajar tentang cara kerja model bahasa."],
  },
  umur: {
    user: ["umur kamu berapa?", "kamu umur berapa?", "kamu lahir kapan?"],
    bot: ["Saya program komputer, jadi saya tidak punya umur seperti manusia."],
  },
  bahasa: {
    user: ["kamu bisa bahasa inggris?", "can you speak english?", "bisa bahasa lain?"],
    bot: ["Saya dilatih dengan bahasa Indonesia, jadi saya paling baik menjawab dalam bahasa Indonesia."],
  },
  pamit: {
    sapaan: true,
    user: ["dadah", "sampai jumpa", "bye", "selamat tinggal", "sudah dulu ya", "oke cukup, terima kasih", "aku pergi dulu ya", "sampai nanti"],
    bot: ["Sampai jumpa! Semoga hari Anda menyenangkan.", "Terima kasih sudah mampir. Sampai jumpa lagi.", "Sampai jumpa, jaga kesehatan ya."],
  },
  maaf: {
    user: ["maaf", "maaf mengganggu", "maaf ya, salah ketik", "maaf saya bertanya lagi", "mohon maaf"],
    bot: ["Tidak apa-apa. Ada yang bisa saya bantu?", "Tidak masalah, silakan bertanya kapan saja."],
  },
  oke: {
    user: ["oke", "ok", "baik", "siap", "sip", "oke deh", "baiklah", "mengerti", "hmm", "hmmm", "oh begitu", "oh gitu", "ok sip", "sip deh"],
    bot: ["Baik. Jika ada pertanyaan lain, saya siap membantu.", "Oke! Silakan bertanya lagi kapan saja."],
  },
  pujian: {
    user: ["kamu pintar", "bagus sekali", "keren", "mantap", "jawabanmu membantu", "hebat", "lucu juga kamu", "kamu lucu", "wkwk", "haha", "pintar juga kamu"],
    bot: ["Terima kasih atas pujiannya! Saya akan terus berusaha membantu.", "Terima kasih, senang mendengarnya."],
  },
  bingung: {
    user: ["aku bingung", "saya tidak mengerti", "jawabanmu kurang jelas", "maksudnya gimana?", "bisa diulang?"],
    bot: [
      "Maaf jika kurang jelas. Coba sampaikan pertanyaan dengan kata lain, saya akan berusaha membantu.",
      "Baik, coba jelaskan bagian mana yang membingungkan, nanti saya bantu sebisanya.",
    ],
  },
  luarKemampuan: {
    user: [
      "cuaca hari ini bagaimana?", "siapa juara piala dunia?", "berapa harga emas hari ini?", "rekomendasi film dong",
      "resep nasi goreng apa?", "skor pertandingan semalam berapa?", "kapan libur sekolah?", "harga saham hari ini?",
    ],
    bot: [
      "Maaf, saya tidak punya informasi tentang itu. Saya lebih bisa membantu soal aturan pemerintah.",
      "Maaf, pertanyaan itu di luar pengetahuan saya. Silakan tanyakan hal yang berkaitan dengan aturan pemerintah.",
    ],
  },
  lelucon: {
    user: ["ceritakan lelucon", "kasih tebak-tebakan dong", "buat aku tertawa", "ada cerita lucu?"],
    bot: [
      "Kenapa komputer jarang kelaparan? Karena sudah punya banyak byte.",
      "Tebak-tebakan: hewan apa yang paling hormat? Hormat-ta. Semoga bisa membuat Anda tersenyum.",
    ],
  },
  sedih: {
    user: ["aku lagi sedih", "capek banget kerja hari ini", "lagi banyak pikiran", "aku stres", "aku bosan", "lagi bosan nih", "bete", "lagi galau", "aku capek", "capek", "lelah sekali", "aku lelah"],
    bot: [
      "Saya ikut prihatin. Semoga keadaan Anda segera membaik. Jangan lupa istirahat ya.",
      "Wajar merasa lelah. Coba luangkan waktu sejenak untuk beristirahat. Semoga harimu lebih baik.",
    ],
  },
  tes: {
    user: ["tes", "test", "tes tes", "ping", "p", "halo tes"],
    bot: ["Ya, saya di sini. Ada yang bisa saya bantu?", "Pesan diterima. Silakan sampaikan pertanyaan Anda."],
  },
  mauTanya: {
    sapaan: true,
    user: ["saya mau tanya", "boleh tanya?", "mau tanya dong", "ada yang mau saya tanyakan", "saya ingin bertanya soal aturan", "boleh minta tolong?", "bisa bantu saya?", "tolong bantu saya", "minta bantuan dong"],
    bot: ["Tentu, silakan. Apa yang ingin ditanyakan?", "Boleh sekali. Silakan sampaikan pertanyaan Anda."],
  },
};

/** Urutan niat yang wajar untuk percakapan beberapa giliran. */
const ALUR: string[][] = [
  ["salam", "kabar", "kabarBaik"],
  ["salam", "kabar", "kabarBuruk"],
  ["pagi", "mauTanya"],
  ["salam", "identitas", "kemampuan"],
  ["salam", "mauTanya", "oke", "terimaKasih", "pamit"],
  ["salamIslam", "kabar", "kabarBaik", "pamit"],
  ["malam", "identitas", "robot"],
  ["salam", "luarKemampuan", "oke"],
  ["sore", "sedih", "terimaKasih"],
  ["salam", "lelucon", "pujian"],
  ["tes", "salam", "kemampuan", "terimaKasih"],
  ["siang", "bingung", "maaf"],
  ["identitas", "pembuat", "umur"],
  ["kemampuan", "terimaKasih", "pamit"],
];

const SAPAAN = ["kak", "pak", "bu", "min", "bang", "mas", "mbak"];

export function pilih<T>(rng: Rng, a: T[]): T {
  return a[rng.int(a.length)];
}

/** Variasi kalimat pengguna: sapaan, huruf besar/kecil, tanda baca, salah ketik ringan. */
export function variasi(rng: Rng, s: string, bolehSapaan: boolean): string {
  let t = s;
  if (bolehSapaan && rng.next() < 0.3) t = t.replace(/[?!.]?$/, (m) => ` ${pilih(rng, SAPAAN)}${m}`);
  const r = rng.next();
  if (r < 0.35) t = t.charAt(0).toUpperCase() + t.slice(1);
  else if (r < 0.45) t = t.toUpperCase();
  const tanda = rng.next();
  if (tanda < 0.2) t = t.replace(/[?!.]$/, "");
  else if (tanda < 0.35 && !/[?!.]$/.test(t)) t += "!";
  else if (tanda < 0.45 && !/[?!.]$/.test(t)) t += ".";
  if (rng.next() < 0.08 && t.length > 4) { // salah ketik: hapus satu huruf di tengah
    const i = 1 + rng.int(t.length - 2);
    t = t.slice(0, i) + t.slice(i + 1);
  }
  return t;
}

/**
 * Niat yang memakai NAMA pengguna ({N}). Nama diambil acak dari daftar, sehingga model belajar
 * MENYALIN nama dari giliran perkenalan sebelumnya (attention melihat ke belakang), bukan menghafal satu nama.
 * Chatbot memakai pola yang sama untuk memori: nama yang disimpan pengguna diletakkan sebagai
 * giliran perkenalan di awal prompt.
 */
export const NIAT_NAMA: Record<string, Niat> = {
  perkenalan: {
    user: ["nama saya {N}", "namaku {N}", "panggil aku {N}", "perkenalkan, saya {N}", "halo, saya {N}", "nama aku {N}"],
    bot: ["Senang berkenalan dengan Anda, {N}! Ada yang bisa saya bantu?", "Halo, {N}! Senang berkenalan dengan Anda."],
  },
  tanyaNama: {
    user: ["siapa nama saya?", "kamu ingat nama saya?", "namaku siapa?", "tahu namaku?", "masih ingat namaku?"],
    bot: ["Nama Anda {N}.", "Tentu, nama Anda {N}."],
  },
  salamNama: {
    user: ["halo", "hai", "selamat pagi", "selamat malam", "halo lagi", "hai, aku kembali"],
    bot: ["Halo, {N}! Ada yang bisa saya bantu?", "Hai, {N}! Senang bertemu lagi."],
  },
  terimaKasihNama: {
    user: ["terima kasih", "makasih ya", "terima kasih banyak"],
    bot: ["Sama-sama, {N}! Senang bisa membantu.", "Sama-sama, {N}."],
  },
  pamitNama: {
    user: ["dadah", "sampai jumpa", "aku pergi dulu ya"],
    bot: ["Sampai jumpa, {N}! Semoga hari Anda menyenangkan.", "Sampai jumpa lagi, {N}."],
  },
};
const TANPA_NAMA: Niat = {
  user: NIAT_NAMA.tanyaNama.user,
  bot: ["Maaf, saya belum tahu nama Anda. Anda bisa memberi tahu saya, misalnya: nama saya Budi.", "Saya belum tahu nama Anda. Siapa nama Anda?"],
};
export const NAMA_ORANG = [
  "Budi", "Siti", "Irun", "Andi", "Rina", "Dewi", "Agus", "Putri", "Rizky", "Fajar", "Ayu", "Bayu", "Dian", "Eka", "Fitri",
  "Gita", "Hadi", "Indra", "Joko", "Kartika", "Lina", "Maya", "Nanda", "Rudi", "Sari", "Tono", "Wati", "Yudi", "Zahra", "Bima",
  "Citra", "Dimas", "Rahma", "Arif", "Nisa", "Taufik", "Wulan", "Yoga", "Reza", "Intan", "Hendra", "Lestari", "Galih", "Mega",
];
/** Niat yang boleh muncul di tengah percakapan acak (tidak bergantung pada giliran sebelumnya). */
export const NIAT_BEBAS = [
  "salam", "pagi", "siang", "sore", "malam", "kabar", "terimaKasih", "identitas", "robot", "kemampuan", "pembuat", "umur",
  "bahasa", "pamit", "maaf", "oke", "pujian", "bingung", "luarKemampuan", "lelucon", "sedih", "tes", "mauTanya",
];

export function buatPercakapan(jumlahTunggal = 1200, jumlahAlur = 600, seed = 1): { user?: string; bot?: string }[][] {
  const rng = new Rng(seed);
  const nama = Object.keys(NIAT);
  const giliranDari = (ni: Niat, orang = "") => {
    const ganti = (s: string) => s.replaceAll("{N}", orang);
    return [{ user: ganti(variasi(rng, pilih(rng, ni.user), !!ni.sapaan)) }, { bot: ganti(pilih(rng, ni.bot)) }];
  };
  const giliran = (n: string) => giliranDari(NIAT[n]);
  const hasil: { user?: string; bot?: string }[][] = [];
  for (let i = 0; i < jumlahTunggal; i++) hasil.push(giliran(nama[i % nama.length]));
  for (let i = 0; i < jumlahAlur; i++) hasil.push(pilih(rng, ALUR).flatMap(giliran));

  // Rangkaian acak 2–4 niat: model belajar membalas pesan TERAKHIR apa pun riwayatnya
  for (let i = 0; i < 700; i++) {
    const n = 2 + rng.int(3);
    hasil.push(Array.from({ length: n }, () => pilih(rng, NIAT_BEBAS)).flatMap(giliran));
  }
  // Percakapan dengan nama: perkenalan, (0–2 niat lain), lalu niat yang memakai nama
  for (let i = 0; i < 900; i++) {
    const orang = pilih(rng, NAMA_ORANG);
    const tengah = Array.from({ length: rng.int(3) }, () => pilih(rng, NIAT_BEBAS)).flatMap(giliran);
    const akhir = pilih(rng, ["tanyaNama", "salamNama", "terimaKasihNama", "pamitNama", "tanyaNama"]);
    hasil.push([...giliranDari(NIAT_NAMA.perkenalan, orang), ...tengah, ...giliranDari(NIAT_NAMA[akhir], orang)]);
  }
  // Ditanya nama tanpa pernah berkenalan -> jujur belum tahu
  for (let i = 0; i < 120; i++) hasil.push([...Array.from({ length: rng.int(2) }, () => pilih(rng, NIAT_BEBAS)).flatMap(giliran), ...giliranDari(TANPA_NAMA)]);
  // acak urutan agar setiap blok val berisi campuran niat
  for (let i = hasil.length - 1; i > 0; i--) { const j = rng.int(i + 1); [hasil[i], hasil[j]] = [hasil[j], hasil[i]]; }
  return hasil;
}

// Dataset lengkap (obrolan biasa + percakapan berkonteks) dibuat oleh src/commands/make-chat-data.ts.

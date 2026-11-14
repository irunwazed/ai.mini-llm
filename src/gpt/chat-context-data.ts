/**
 * DATA PERCAKAPAN BERKONTEKS untuk melatih GPT memakai informasi dari giliran SEBELUMNYA.
 *
 * Data obrolan lama (chat-data.ts) hampir semuanya satu niat per giliran: balasan tidak bergantung
 * pada riwayat. Di sini setiap percakapan sengaja dibuat sehingga jawaban terakhir HANYA bisa benar
 * jika model melihat ke belakang:
 *   ingat fakta     "namaku Sari, aku tinggal di Medan" … "aku tinggal di mana?" -> "Anda tinggal di Medan."
 *   pilih fakta     beberapa fakta disebut, yang ditanya hanya satu
 *   kata ganti      "aku punya kucing namanya Oyen" -> "dia suka ikan" -> "dia suka makan apa?"
 *   koreksi         "aku dari Bogor" -> "eh salah, maksudku Bandung" -> "aku dari mana?" -> Bandung
 *   topik lanjutan  "aku lagi belajar berenang" -> "susah nggak?" -> jawaban menyebut berenang
 *   ringkas         "tadi kita ngobrol apa?"
 *   suasana         "aku sedih karena gagal ujian" … "sekarang sudah lega" -> merujuk alasan tadi
 *   belum tahu      ditanya fakta yang belum pernah disebut -> jujur "belum memberi tahu"
 *   kartu konteks   <|konteks|> nama: …; tinggal: …  (fakta dari memori/sesi lama) -> dipakai menjawab
 *
 * NILAI SLOT (nama, kota, pekerjaan, …) DIBAGI DUA: "latih" dan "uji". Data latih hanya memakai nilai
 * latih; eval-konteks memakai nilai uji yang belum pernah dilihat model. Jika model menjawab benar
 * dengan nilai uji, berarti ia benar-benar MENYALIN dari konteks, bukan menghafal pasangan tanya-jawab.
 */
import { NAMA_ORANG, NIAT, NIAT_BEBAS, pilih, variasi } from "./chat-data.js";
import { Rng } from "./random.js";

export type GiliranKonteks = { user?: string; bot?: string; konteks?: string };
export interface PercakapanKonteks { skenario: string; turns: GiliranKonteks[]; harapan: string }

const NAMA = [
  "Adi", "Aisyah", "Aldi", "Amalia", "Anisa", "Anton", "Arini", "Asep", "Bagas", "Bella", "Benny", "Bunga", "Cahya", "Danang", "Desi",
  "Dodi", "Elsa", "Endah", "Erwin", "Farah", "Fikri", "Gilang", "Hana", "Haris", "Ikhsan", "Ilham", "Jihan", "Kevin", "Laila", "Lutfi",
  "Melati", "Nadia", "Nugroho", "Okta", "Pandu", "Prita", "Raka", "Ratna", "Salsa", "Satria", "Tiara", "Umar", "Vina", "Wahyu",
  "Yosef", "Yuni", "Zaki", "Annisa", "Bambang", "Cindy", "Dedi", "Evi", "Faisal", "Gina", "Hesti", "Ivan", "Jaka", "Kirana",
  "Larasati", "Mirna", "Nina", "Oki", "Puspita", "Rangga", "Septi", "Toni", "Ujang", "Vera", "Wisnu", "Yanti", "Zul", "Aulia",
  "Bintang", "Damar", "Elina", "Fadli", "Gabriel", "Hilda", "Irfan", "Jessica", "Krisna", "Lukman", "Meiliana", "Novi", "Opik",
].filter((n) => !NAMA_ORANG.includes(n));
const KOTA = [
  "Medan", "Bandung", "Surabaya", "Makassar", "Palembang", "Semarang", "Yogyakarta", "Malang", "Bogor", "Padang", "Pekanbaru", "Banjarmasin",
  "Pontianak", "Manado", "Denpasar", "Mataram", "Kupang", "Ambon", "Jayapura", "Balikpapan", "Samarinda", "Jambi", "Bengkulu", "Lampung",
  "Serang", "Cirebon", "Tegal", "Solo", "Kediri", "Jember", "Banyuwangi", "Madiun", "Purwokerto", "Tasikmalaya", "Garut", "Sukabumi",
  "Bekasi", "Depok", "Tangerang", "Batam", "Pangkalpinang", "Palu", "Kendari", "Gorontalo", "Ternate", "Sorong", "Aceh", "Sabang",
  "Bukittinggi", "Payakumbuh", "Magelang", "Klaten", "Probolinggo", "Pasuruan", "Blitar", "Lumajang", "Kudus", "Pati", "Rembang", "Cilacap",
];
const KERJA = [
  "guru", "perawat", "dokter", "programmer", "petani", "nelayan", "pedagang", "sopir", "penjahit", "koki", "polisi", "tentara", "pilot",
  "arsitek", "akuntan", "wartawan", "desainer", "montir", "satpam", "bidan", "apoteker", "dosen", "pegawai negeri", "kasir", "barista",
  "fotografer", "penulis", "penyiar radio", "tukang kayu", "insinyur", "pengacara", "notaris", "peneliti", "pustakawan", "pelatih renang",
  "kurir", "admin kantor", "teknisi listrik", "peternak", "pemandu wisata",
];
const HOBI = [
  "memasak", "berenang", "bersepeda", "membaca", "menulis", "melukis", "menyanyi", "bermain gitar", "bermain catur", "fotografi", "berkebun",
  "memancing", "mendaki gunung", "menari", "bermain bulu tangkis", "bermain sepak bola", "lari pagi", "yoga", "menjahit", "merajut",
  "bermain piano", "menonton film", "mengoleksi perangko", "membuat kue", "bermain game", "bermain voli", "panahan", "kaligrafi",
  "origami", "bermain biola", "menyelam", "berkemah", "senam", "karate", "silat", "bermain drum",
];
const MAKANAN = [
  "rendang", "sate ayam", "bakso", "soto", "nasi goreng", "mie ayam", "gado-gado", "pempek", "rawon", "gudeg", "nasi uduk", "ayam geprek",
  "martabak", "pecel lele", "sop buntut", "nasi padang", "seblak", "batagor", "siomay", "lontong sayur", "coto makassar", "papeda",
  "ikan bakar", "tahu tek", "lumpia", "kerak telor", "bubur ayam", "nasi kuning", "karedok", "rujak", "es campur", "klepon",
];
const HEWAN = ["kucing", "anjing", "burung", "kelinci", "hamster", "ikan", "kura-kura", "ayam"];
const NAMA_HEWAN = [
  "Oyen", "Mimi", "Belang", "Kiki", "Bobi", "Moli", "Coco", "Simba", "Luna", "Milo", "Snowy", "Cemong", "Pusi", "Tompel", "Bruno", "Kitty",
  "Leo", "Nala", "Choki", "Ucil", "Gembul", "Mochi", "Kopi", "Susu", "Pipi", "Bolu", "Tiger", "Lulu", "Max", "Rocky", "Bento", "Dodol",
];
const MAKAN_HEWAN = ["ikan", "daging ayam", "wortel", "biji-bijian", "sayur", "sosis", "nasi", "buah pisang", "pakan kering"];
const ALASAN = [
  "gagal ujian", "kehilangan dompet", "bertengkar dengan teman", "gagal wawancara kerja", "rindu keluarga", "nilai rapor turun",
  "ponselku rusak", "dimarahi atasan", "hujan terus jadi tidak bisa pergi", "batal liburan", "sakit gigi", "ketinggalan kereta",
  "proyek kantor ditolak", "uang jajan habis", "kucingku sakit",
];

/** Bagi daftar menjadi nilai latih (≈ 80%) dan nilai uji (setiap nilai ke-5). */
function bagi<T>(a: T[], uji: boolean): T[] { return a.filter((_, i) => (i % 5 === 4) === uji); }
export function daftarSlot(uji: boolean) {
  return {
    nama: bagi(NAMA, uji), kota: bagi(KOTA, uji), kerja: bagi(KERJA, uji), hobi: bagi(HOBI, uji), makanan: bagi(MAKANAN, uji),
    hewan: HEWAN, namaHewan: bagi(NAMA_HEWAN, uji), makanHewan: MAKAN_HEWAN, alasan: bagi(ALASAN, uji),
  };
}

type Slot = "nama" | "kota" | "kerja" | "hobi" | "makanan";
const KENAL: Record<Slot, { user: string[]; bot: string[]; tanya: string[]; jawab: string[]; belum: string; kartu: string }> = {
  nama: {
    user: ["namaku {v}", "nama saya {v}", "panggil aku {v}", "kenalin, aku {v}"],
    bot: ["Senang berkenalan dengan Anda, {v}!", "Halo, {v}! Senang berkenalan."],
    tanya: ["siapa namaku?", "kamu ingat namaku?", "namaku siapa tadi?"], jawab: ["Nama Anda {v}.", "Tentu, nama Anda {v}."],
    belum: "Anda belum memberi tahu nama Anda.", kartu: "nama",
  },
  kota: {
    user: ["aku tinggal di {v}", "aku dari {v}", "rumahku di {v}", "aku orang {v}"],
    bot: ["Wah, {v}! Kota yang menarik.", "Oh, Anda dari {v}. Senang mendengarnya."],
    tanya: ["aku tinggal di mana?", "aku dari mana tadi?", "kotaku apa?", "kamu ingat aku tinggal di mana?"], jawab: ["Anda tinggal di {v}.", "Anda dari {v}."],
    belum: "Anda belum memberi tahu tempat tinggal Anda.", kartu: "tinggal",
  },
  kerja: {
    user: ["aku kerja sebagai {v}", "pekerjaanku {v}", "aku seorang {v}", "aku bekerja jadi {v}"],
    bot: ["Pekerjaan sebagai {v} pasti menarik.", "Wah, {v}! Terima kasih sudah berbagi."],
    tanya: ["pekerjaanku apa?", "aku kerja apa?", "kamu ingat pekerjaanku?"], jawab: ["Anda bekerja sebagai {v}.", "Pekerjaan Anda {v}."],
    belum: "Anda belum memberi tahu pekerjaan Anda.", kartu: "pekerjaan",
  },
  hobi: {
    user: ["hobiku {v}", "aku suka {v}", "kegemaranku {v}", "di waktu luang aku suka {v}"],
    bot: ["{V} hobi yang bagus!", "Wah, {v} itu menyenangkan."],
    tanya: ["hobiku apa?", "tadi aku bilang suka apa?", "kegemaranku apa?"], jawab: ["Hobi Anda {v}.", "Anda suka {v}."],
    belum: "Anda belum memberi tahu hobi Anda.", kartu: "hobi",
  },
  makanan: {
    user: ["makanan favoritku {v}", "aku paling suka makan {v}", "aku doyan banget {v}"],
    bot: ["{V} memang enak!", "Wah, {v}! Pilihan yang lezat."],
    tanya: ["makanan favoritku apa?", "aku paling suka makan apa?"], jawab: ["Makanan favorit Anda {v}.", "Anda paling suka {v}."],
    belum: "Anda belum memberi tahu makanan favorit Anda.", kartu: "makanan favorit",
  },
};
const ganti = (s: string, v: string) => s.replaceAll("{v}", v).replaceAll("{V}", v.charAt(0).toUpperCase() + v.slice(1));

export function buatPercakapanKonteks(jumlah: number, uji: boolean, seed = 7): PercakapanKonteks[] {
  const rng = new Rng(seed);
  const s = daftarSlot(uji);
  const u = (t: string) => ({ user: variasi(rng, t, false) });
  const b = (t: string) => ({ bot: t });
  /** 0–n giliran obrolan biasa (pengalih), supaya fakta tidak selalu tepat di giliran sebelumnya. */
  const selingan = (n: number) => Array.from({ length: rng.int(n + 1) }, () => {
    const ni = NIAT[pilih(rng, NIAT_BEBAS.filter((x) => !["sedih", "pamit", "tes"].includes(x)))];
    return [u(pilih(rng, ni.user)), b(pilih(rng, ni.bot))];
  }).flat();
  const nilai = (k: Slot) => pilih(rng, s[k]);
  const kenal = (k: Slot, v: string) => [u(ganti(pilih(rng, KENAL[k].user), v)), b(ganti(pilih(rng, KENAL[k].bot), v))];

  const skenario: [string, number, () => PercakapanKonteks][] = [
    ["ingat fakta", 3, () => {
      const k = pilih(rng, Object.keys(KENAL) as Slot[]), v = nilai(k);
      const jawab = ganti(pilih(rng, KENAL[k].jawab), v);
      return { skenario: "ingat fakta", harapan: v, turns: [...selingan(1), ...kenal(k, v), ...selingan(3), u(pilih(rng, KENAL[k].tanya)), b(jawab)] };
    }],
    ["pilih fakta", 3, () => {
      const kunci = (Object.keys(KENAL) as Slot[]).sort(() => rng.next() - 0.5).slice(0, 2 + rng.int(2));
      const v = Object.fromEntries(kunci.map((k) => [k, nilai(k)])) as Record<Slot, string>;
      const ditanya = pilih(rng, kunci);
      const turns = kunci.flatMap((k) => [...kenal(k, v[k]), ...selingan(1)]);
      return { skenario: "pilih fakta", harapan: v[ditanya], turns: [...turns, u(pilih(rng, KENAL[ditanya].tanya)), b(ganti(pilih(rng, KENAL[ditanya].jawab), v[ditanya]))] };
    }],
    ["kata ganti", 2, () => {
      const h = pilih(rng, s.hewan), nh = pilih(rng, s.namaHewan), m = pilih(rng, s.makanHewan);
      const awal = [u(`aku punya ${h} namanya ${nh}`), b(`Lucu sekali! ${nh} pasti menggemaskan.`), u(`dia suka makan ${m}`), b(`Wah, ${nh} suka ${m} ya.`)];
      const tanya = pilih(rng, [
        [`siapa nama ${h}ku?`, `Nama ${h} Anda ${nh}.`, nh], [`dia suka makan apa?`, `${nh} suka makan ${m}.`, m],
        [`hewan peliharaanku apa?`, `Anda punya ${h} bernama ${nh}.`, h], [`${nh} itu hewan apa?`, `${nh} adalah ${h} Anda.`, h],
      ] as [string, string, string][]);
      return { skenario: "kata ganti", harapan: tanya[2], turns: [...awal, ...selingan(2), u(tanya[0]), b(tanya[1])] };
    }],
    ["koreksi", 1, () => {
      const k1 = nilai("kota"); let k2 = nilai("kota"); while (k2 === k1) k2 = nilai("kota");
      return {
        skenario: "koreksi", harapan: k2, turns: [
          u(`aku tinggal di ${k1}`), b(`Wah, ${k1}! Kota yang menarik.`),
          u(pilih(rng, [`eh salah, maksudku ${k2}`, `maaf, bukan ${k1}, tapi ${k2}`, `ralat, aku tinggal di ${k2}`])), b(`Baik, saya catat Anda tinggal di ${k2}.`),
          ...selingan(2), u(pilih(rng, KENAL.kota.tanya)), b(`Anda tinggal di ${k2}.`),
        ],
      };
    }],
    ["topik lanjutan", 2, () => {
      const h = nilai("hobi");
      const lanjut = pilih(rng, [
        ["susah nggak?", `${ganti("{V}", h)} memang butuh latihan di awal, tapi lama-lama akan terbiasa.`],
        ["gimana cara mulainya?", `Untuk mulai ${h}, coba pelajari dasarnya dulu dan berlatih sedikit setiap hari.`],
        ["butuh berapa lama?", `Lama belajar ${h} berbeda-beda untuk setiap orang, yang penting rutin.`],
        ["ada tipsnya?", `Tips ${h}: sabar, rutin berlatih, dan belajar dari yang sudah mahir.`],
        ["kalau untuk pemula gimana?", `Untuk pemula, ${h} sebaiknya dimulai dari hal yang paling sederhana.`],
      ]);
      return { skenario: "topik lanjutan", harapan: h, turns: [...selingan(1), u(`aku lagi belajar ${h}`), b(`Bagus! Belajar ${h} memang menyenangkan.`), u(lanjut[0]), b(lanjut[1])] };
    }],
    ["ringkas", 1, () => {
      const pilihan = pilih(rng, [["hobi", "hobi Anda, {v}"], ["makanan", "makanan favorit Anda, {v}"], ["kota", "kota Anda, {v}"], ["kerja", "pekerjaan Anda sebagai {v}"]] as [Slot, string][]);
      const v = nilai(pilihan[0]);
      return { skenario: "ringkas", harapan: v, turns: [...kenal(pilihan[0], v), u(pilih(rng, ["tadi kita ngobrol apa?", "kita lagi bahas apa?", "tadi kita bicara soal apa?"])), b(`Tadi kita membahas ${ganti(pilihan[1], v)}.`)] };
    }],
    ["suasana", 1, () => {
      const a = pilih(rng, s.alasan);
      return {
        skenario: "suasana", harapan: a, turns: [
          u(`aku sedih karena ${a}`), b(`Saya ikut sedih soal ${a}. Semoga cepat membaik ya.`), ...selingan(1),
          u(pilih(rng, ["sekarang sudah agak lega", "aku sudah merasa lebih baik", "udah mendingan sekarang"])),
          b(`Syukurlah. Semoga urusan ${a} juga cepat selesai.`),
        ],
      };
    }],
    ["belum tahu", 1, () => {
      const kunci = Object.keys(KENAL) as Slot[];
      const disebut = pilih(rng, kunci); let ditanya = pilih(rng, kunci); while (ditanya === disebut) ditanya = pilih(rng, kunci);
      return { skenario: "belum tahu", harapan: "belum", turns: [...kenal(disebut, nilai(disebut)), ...selingan(1), u(pilih(rng, KENAL[ditanya].tanya)), b(KENAL[ditanya].belum)] };
    }],
    ["kartu konteks", 2, () => {
      // fakta hanya ada di kartu (mis. dari memori atau sesi lama yang sudah terpotong)
      const kunci = (Object.keys(KENAL) as Slot[]).sort(() => rng.next() - 0.5).slice(0, 1 + rng.int(3));
      const v = Object.fromEntries(kunci.map((k) => [k, nilai(k)])) as Record<Slot, string>;
      const kartu = kunci.map((k) => `${KENAL[k].kartu}: ${v[k]}`).join("; ");
      const ditanya = pilih(rng, kunci);
      const sapa = kunci.includes("nama") && rng.next() < 0.3;
      const akhir = sapa ? [u(pilih(rng, ["halo", "hai", "selamat pagi"])), b(`Halo, ${v.nama}! Ada yang bisa saya bantu?`)]
        : [u(pilih(rng, KENAL[ditanya].tanya)), b(ganti(pilih(rng, KENAL[ditanya].jawab), v[ditanya]))];
      return { skenario: "kartu konteks", harapan: sapa ? v.nama : v[ditanya], turns: [{ konteks: kartu }, ...selingan(2), ...akhir] };
    }],
  ];
  const bobot = skenario.reduce((a, x) => a + x[1], 0);
  const hasil: PercakapanKonteks[] = [];
  for (let i = 0; i < jumlah; i++) {
    let r = rng.next() * bobot;
    const sk = skenario.find((x) => (r -= x[1]) < 0) ?? skenario[0];
    hasil.push(sk[2]());
  }
  return hasil;
}

# LLM dari Nol + Chatbot Aturan Pemerintah + GPT Mini

> **Panduan praktis** per langkah (ambil data → siapkan → latih → evaluasi → jalankan, lengkap dengan lokasi setiap file): lihat [PANDUAN.md](PANDUAN.md). Semua langkah juga bisa dijalankan dan dipantau dari web: `bun run web` lalu buka http://localhost:3000/latih (progress bar, grafik loss, log lengkap, berhenti aman, upload file). Di terminal: `bun run status -- --ikuti`; Ctrl+C menghentikan dengan aman (lanjutkan dengan `--resume`).

Proyek belajar tiga jalur, semuanya **Bun + TypeScript tanpa library ML**:

1. **`lessons/`** belajar model bahasa dari nol secara matematika: bigram, n-gram, BPE, attention, GPT mini.
2. **Chatbot aturan** menjawab pertanyaan dari ribuan dokumen aturan (PDF, DOCX, TXT, MD). Jawabannya berupa kesimpulan, rincian, dan **dasar hukum** (peraturan, pasal, ayat, file, halaman). Setiap fakta diambil dari kalimat asli dokumen, jadi chatbot tidak mengarang.
3. **GPT** (decoder-only transformer) yang dilatih dari nol dan **menghasilkan teks** token demi token, mengikuti diagram di bawah.

Nama file dan folder memakai bahasa Inggris. Teks yang dipelajari model dan percakapan chatbot berbahasa Indonesia.

## Struktur folder

```
llm/
├── documents/          ← TARUH FILE ATURAN DI SINI (.pdf .docx .txt .md, boleh bersubfolder)
│   ├── samples/        5 dokumen contoh FIKTIF, dirujuk soal evaluasi (jangan dihapus)
│   ├── jdih/           unduhan `bun run fetch-jdih`
│   └── upload/         upload dari halaman web /latih
├── data/
│   ├── mentah/         LANGKAH 1 — data mentah
│   │   ├── wikipedia/  wiki-id.txt + sumber.txt        (`bun run fetch-data`)
│   │   ├── teks/       teks bahasa Indonesia milik Anda (upload / salin)
│   │   ├── obrolan/    chat-id.jsonl                   (`bun run make-chat-data`)
│   │   └── pembaca/    reader-id.jsonl                 (`bun run make-reader-data`)
│   ├── siap/<dataset>/ LANGKAH 2 — tokenizer + token siap latih (`bun run siapkan-data`)
│   ├── synonyms.txt    kamus padanan kata (tanpa training ulang)
│   ├── slang.txt       kamus bahasa santai → baku
│   └── corpus.txt      teks kecil untuk pelajaran
├── model/              LANGKAH 3 — indeks chatbot aturan (`bun run train`)
├── model-wiki/         LANGKAH 3 — indeks Wikipedia (`bun run index-wiki`)
├── gpt-models/         LANGKAH 3 — satu folder per model GPT (`bun run train-gpt` / `train-reader`)
├── progres/            progres + log setiap proses (`bun run status`, halaman /latih)
├── evaluation/         LANGKAH 4 — soal uji (questions, conversations, reasoning)
├── lessons/            01_bigram, 02_ngram, 03_bpe, 04_attention, 05_mini_gpt
└── src/
    ├── paths.ts        SEMUA lokasi data di satu tempat
    ├── readers/        membaca PDF, DOCX, TXT/MD (dengan cache)
    ├── text/           pembersih header/OCR, tokenisasi, stemmer, sinonim, pemotong pasal
    ├── training/       metadata & perubahan aturan, glosarium & singkatan, word2vec
    ├── search/         BM25 + pencarian hibrida + internet (API resmi)
    ├── answer/         jenis pertanyaan, pemecah kalimat, penyusun jawaban, pengutip kalimat internet
    ├── gpt/            GPT dari nol: bpe, layers, model, optimizer, sampling, worker, dataset, checkpoint,
    │                   prepared (dataset siap latih), progress (Pelacak), reader (GPT pembaca), data obrolan/konteks
    ├── rag/            RAG: konteks percakapan (dialog), ambil konteks (retrieve), baca + verifikasi (pipeline)
    ├── agent/          mode agent: semua mode + pencarian internet
    ├── reasoning/      agent penalaran (kalkulator, kalender, barisan, …) + pemeriksa relevansi
    ├── web/            index.html (chat), latih.html (data & training), jobs.ts (proses dari web)
    ├── model.ts        simpan / muat model chatbot
    └── commands/       satu file per perintah `bun run …`
```

## Perintah

Sekali saja di awal:

```
bun install
```

**Chatbot aturan**

| Tujuan | Perintah |
|---|---|
| Training dari folder `documents/` | `bun run train` |
| Training dari folder lain | `bun run train -- --dokumen D:/aturan` |
| Unduh peraturan JDIH BKN ke `documents/jdih/` | `bun run fetch-jdih` (crawler otomatis; ulangi kapan saja — yang sudah ada tidak diunduh lagi) |
| Tanya satu pertanyaan | `bun run search -- "berapa hari cuti tahunan?"` |
| Chatbot interaktif | `bun run chat` |
| Chatbot mode agent (semua mode + internet) | `bun run chat -- --agent` (atau ketik `:agent` di dalam chat) |
| Mode agent tanpa internet | `bun run chat -- --agent --tanpa-internet` |
| Ukur akurasi jawaban aturan | `bun run eval` |
| Uji percakapan "nyambung" | `bun run eval-chat` |
| Uji percakapan lewat mode agent (tanpa internet) | `bun run eval-chat -- --agent` |
| Chatbot dengan RAG (GPT pembaca) | `bun run chat -- --rag` (atau ketik `:rag`) |
| Ukur akurasi + RAG (berapa jawaban pembaca diterima & benar) | `bun run eval -- --rag` |
| Uji percakapan lewat mode RAG | `bun run eval-chat -- --rag` |
| Uji GPT pembaca (EM/F1 di artikel val) | `bun run eval-reader` |
| Uji soal logika & matematika | `bun run eval-reasoning` |

**GPT, alur lengkap seperti ChatGPT (dalam versi mini)**: ambil data → siapkan → latih → evaluasi

```
bun run test-gpt                                                         # 0. pastikan semua turunan benar
bun run fetch-data                                                       # 1. unduh ±10 MB Wikipedia -> data/mentah/wikipedia/
bun run siapkan-data -- --sumber wikipedia --nama wiki-v1 --huruf-asli   # 2. tokenizer + token -> data/siap/wiki-v1/
bun run train-gpt -- --dataset wiki-v1 --name wiki                       # 3. pretraining ±45 menit -> gpt-models/wiki/
bun run generate -- --name wiki "Indonesia adalah"                       #    coba hasilkan teks
bun run make-chat-data                                                   # 1. data percakapan -> data/mentah/obrolan/
bun run siapkan-data -- --sumber obrolan --nama obrolan-v1 --tokenizer-dari wiki   # 2. pakai tokenizer model wiki
bun run train-gpt -- --dataset obrolan-v1 --from wiki --name chat --steps 1500     # 3. fine-tuning chat ±10 menit
bun run chat-gpt                                                         # 5. ngobrol dengan GPT
bun run index-wiki                                                       # 3. indeks Wikipedia untuk pertanyaan fakta
bun run chat                                                             # 5. chatbot lengkap: aturan + fakta + obrolan
bun run web                                                              # 5. versi web -> http://localhost:3000 (+ /latih)
bun run make-reader-data                                                 # 1. data latih GPT pembaca -> data/mentah/pembaca/
bun run train-reader -- --from wiki --name reader --steps 2400           # 3. latih GPT pembaca untuk RAG (±50 menit)
bun run eval-reader                                                      # 4. EM / F1 / tolak benar di artikel val
bun run chat -- --rag                                                    # 5. chatbot dengan RAG (atau mode "RAG" di web)
```

**Penting saat training:**

- **Jangan jalankan dua training dengan `--name` yang sama sekaligus.** Keduanya menulis ke folder yang sama, sehingga checkpoint saling menimpa. Keduanya juga berebut CPU, sehingga masing-masing jauh lebih lambat.
- **Cegah komputer tidur.** Jika komputer tidur di tengah training berbasis `--minutes`, pengukuran kecepatan menjadi salah dan jumlah langkahnya terlalu sedikit. Untuk fine-tuning chat, `--steps 1500` lebih aman.

| Tujuan | Perintah |
|---|---|
| Lanjutkan model yang sama | `bun run train-gpt -- --dataset wiki-v1 --name wiki --resume --minutes 30` |
| Lanjutkan model dengan **data baru** | `bun run siapkan-data -- --sumber wikipedia,teks --nama wiki-v2 --tokenizer-dari wiki`, lalu `bun run train-gpt -- --dataset wiki-v2 --name wiki --resume` |
| Latih dari dokumen aturan | `bun run siapkan-data -- --sumber aturan --nama aturan-v1`, lalu `bun run train-gpt -- --dataset aturan-v1 --name rules` |
| Gabung beberapa sumber | `--sumber wikipedia,teks,aturan` (atau path file/folder) |
| Pindahkan data dari struktur lama | `bun run migrasi-data` |

`--resume`/`--from` menolak dataset yang tokenizernya berbeda dari model (sidik tokenizer di `data/siap/<dataset>/info.json`). Cara lama `--data <file>` (tokenisasi di memori) masih bisa dipakai.

**Lain-lain**: `bun run typecheck`, dan pelajaran `bun run lesson:1` sampai `bun run lesson:5`.

Perintah di dalam chatbot:

```
:ingat teks   simpan catatan ke memori, mis. :ingat nama saya Irun
:memori       lihat semua catatan memori
:lupa N       hapus catatan nomor N   (:lupa semua = hapus semua)
:buka N       tampilkan teks lengkap sumber nomor N
:gpt teks     lanjutkan teks dengan model GPT (karangan model, BUKAN dasar hukum)
:proses       tampilkan/sembunyikan cara kerja
:export       ekspor percakapan untuk disalin (juga disimpan di exports/)
:baru         mulai percakapan baru (memori tetap tersimpan)
:bantuan      daftar perintah
:keluar       berhenti
```

### Riwayat percakapan

Chatbot mengingat percakapan yang sedang berjalan, sehingga pertanyaan lanjutan bisa singkat:

| Anda menulis | Dipahami sebagai |
|---|---|
| "siapa Hö'elün?" lalu "kapan **dia** lahir?" | Pertanyaan tentang Hö'elün (subjek sebelumnya dipakai untuk mencari) |
| "apa itu virus?" lalu "berapa ukuran**nya**?" | Ukuran virus |
| "berapa hari cuti tahunan?" lalu "**kalau** cuti besar?" | Berapa lama cuti besar |
| "**lebih detail**" atau "lanjutkan" | Teks lengkap sumber jawaban sebelumnya |

GPT obrolan juga menerima 2 giliran obrolan terakhir. Model ini dilatih dengan percakapan beberapa giliran yang urutannya acak, agar ia membalas pesan terakhir dengan tepat apa pun riwayatnya.

### Bahasa santai, sinonim, dan pemahaman maksud

Sebelum diproses, pesan dinormalisasi dengan kamus `data/slang.txt`: "gimana kalo gak" menjadi "bagaimana kalau tidak", "brp" menjadi "berapa", "tau" menjadi "tahu". Istilah sehari-hari dipetakan ke istilah aturan lewat `data/synonyms.txt`, termasuk frasa: "cuti hamil" menjadi "cuti melahirkan", "naik pangkat" menjadi "kenaikan pangkat", "vendor" menjadi "penyedia". Kedua kamus boleh Anda tambah sendiri tanpa training ulang; cukup jalankan ulang chat atau web.

Perkenalan dan pertanyaan tentang diri Anda dikenali dengan pola tetap (`src/intent.ts`), bukan ditebak GPT, karena model kecil masih sering salah pada kalimat seperti ini.

Jika ada balasan yang tidak nyambung, tambahkan percakapannya ke `evaluation/conversations.jsonl` (jalur yang diharapkan dan kata yang harus ada), lalu jalankan `bun run eval-chat`. Dengan begitu setiap perbaikan bisa diukur dan tidak merusak kasus yang sudah benar.

### Pemeriksa relevansi (loop "periksa, lalu coba lagi")

Setiap jawaban diperiksa dulu sebelum ditampilkan (`src/reasoning/critic.ts`). Jika gagal, jalur berikutnya dicoba.

| Pertanyaan | Syarat lolos |
|---|---|
| "berapa …" | jawaban memuat angka |
| "apa itu X" | ada kalimat "X adalah/merupakan …" |
| "apakah/bolehkah …" | jawaban memuat ya/tidak/boleh/dilarang/wajib |
| "kapan …" | jawaban memuat keterangan waktu |
| semua pencarian | kalimat utama memuat ≥ 50% kata topik pertanyaan |

- **GPT obrolan memakai best-of-N.** Niat pesan dikenali dulu (salam, terima kasih, curhat, dst.; `src/chat-intent.ts`). GPT lalu membuat hingga 3 balasan dengan suhu yang makin tinggi. Balasan pertama yang niatnya cocok dipakai. Jika semuanya meleset, dipakai balasan contoh dari niat itu.
- **Pertanyaan ambigu dijawab dengan tanya balik.** Contohnya "berapa lama cutinya?" ketika ada beberapa jenis cuti yang sama kuat.
- **Jika tetap tidak ada yang lolos, bot menjawab jujur** "Maaf, saya belum mengerti…" dan tidak mengarang.

Loop ini tidak bisa menjamin 100%, karena pemeriksanya juga aturan perkiraan. Tetapi kesalahan yang terdeteksi diperbaiki otomatis, dan hasilnya terukur dengan `bun run eval-chat`: 86,3% sebelum loop, 100% sesudahnya (sekarang 74/74 giliran, termasuk percakapan lanjutan, RAG, dan agent).

### Agent penalaran (Pikir → Aksi → Hasil → Cek)

Soal hitung dan logika ditangani agent di `src/reasoning/agent.ts`:

1. **Pikir.** Kenali jenis soal dari pola kalimat, lalu pilih alat.
2. **Aksi.** Jalankan alat.
3. **Hasil.** Catat jawaban beserta langkah hitungnya.
4. **Cek.** Masukkan jawaban kembali ke soal, atau hitung ulang dengan cara lain. Jika cek gagal, agent mencoba alat berikutnya.

Semua langkah tampil di panel cara kerja web dan di `:proses` terminal. Mode "Hanya penalaran" di web memaksa jalur ini.

| Alat (dari nol) | Contoh |
|---|---|
| kalkulator (recursive descent, perkalian tersirat) | 8 ÷ 2(2 + 2) = 16, dengan catatan ambiguitas (bisa juga 1) |
| kalender mod 7 | Rabu + 100 hari = Jumat |
| pola barisan (selisih bertingkat / rasio) | 2, 6, 12, 20, 30 → 42 |
| persamaan linear (eliminasi Gauss) | tongkat + bola Rp11.000, selisih Rp10.000 → bola Rp500 |
| laju kerja (proporsi) | 5 mesin / 5 produk / 5 menit → 100 mesin / 100 produk = 5 menit |
| selang waktu | 3 pil tiap 30 menit → 60 menit |
| banding berat (konversi satuan) | 1 kg besi vs 1 kg kapas → sama berat |
| silogisme (himpunan + contoh penyangkal) | semua kucing hewan, beberapa hewan berenang → tidak bisa disimpulkan |
| bank teka-teki | jebakan dan paradoks: sakelar, dua penjaga, 0,999… = 1, dll. |

Batasan yang perlu diketahui:

- **Otak agent ini berbasis aturan.** Pada ChatGPT, yang memilih langkah adalah model bahasa raksasa. GPT kita (±0,46 juta parameter) belum mampu bernalar, jadi tidak bisa menjadi otak agent. Pola agent hanya *memperkuat* otak yang sudah mampu: ia memecah soal, memanggil alat, dan memeriksa ulang, tetapi tidak menciptakan kemampuan bernalar.
- **Bank teka-teki adalah pencarian, bukan penalaran.** Teka-teki jebakan dan paradoks dikenali dari pola kalimat, lalu dijawab dengan penjelasan yang sudah ditulis. Panel cara kerja memberi label "teka-teki dikenal". Soal yang bentuknya jauh berbeda belum tentu dikenali.
- **Alat hitung benar-benar menghitung.** Kategori "variasi angka" di `evaluation/reasoning.jsonl` mengganti angka dan nama soal untuk mengujinya. Hasil saat ini 42/42: 25 soal asli dan 17 variasi.

### Mode agent (loop: pahami → alat → bandingkan → perbaiki → simpulkan)

Pilih **Agent (semua mode + internet)** di web, atau jalankan `bun run chat -- --agent` (atau ketik `:agent`). Kodenya ada di `src/agent/`. Tidak ada layanan AI luar.

```
1. PAHAMI MAKSUD  intent.ts      jenis pesan (obrolan, aturan, fakta, pribadi, hitung), bentuk jawaban yang dicari
                                 (angka+satuan, tanggal, nama, tempat, definisi, daftar), kata kunci, nama diri,
                                 subjek percakapan sebelumnya, sumber paling mungkin
2. PAKAI ALAT     tools.ts       otomatis · aturan · rag · Wikipedia lokal · memori · GPT obrolan · penalaran
                                 · Wikipedia internet (subjek / kata kunci / artikel lengkap) · DuckDuckGo · Brave
3. BANDINGKAN     compare.ts     hasil vs maksud -> skor + daftar KEKURANGAN
4. PERBAIKI       multi-mode.ts  kekurangan -> aksi baru di depan antrian; ulangi 2–4 (maks 8 langkah / 20 detik)
5. SIMPULKAN      synthesize.ts  semua hasil -> kolam bukti -> jawaban disusun agent sendiri
```

| Kekurangan yang terdeteksi | Contoh | Perbaikan otomatis |
|---|---|---|
| nama diri tidak disebut | "kapan Soekarno lahir?" dijawab tentang Romy Soekarno | cari artikel subjeknya, lalu baca artikel lengkap |
| bentuk jawaban salah | "berapa tinggi Rinjani?" dijawab "berdiri tahun 1996" (bukan satuan tinggi) | baca artikel lengkap / sumber lain |
| kata yang ditanya hilang | "lahir" tidak ada di kalimat | coba padanan katanya, lalu cari di internet |
| sumber tidak sesuai | pertanyaan aturan dijawab Wikipedia; fakta dijawab GPT | ganti ke sumber yang sesuai |
| ambigu | "berapa lama cutinya?" | berhenti, tanya balik |

**Kesimpulan disusun agent, bukan diteruskan dari mode lain.** Semua yang terkumpul selama loop masuk ke satu kolam bukti: pasal aturan, potongan Wikipedia, catatan memori, dan **semua** halaman internet yang terambil. Setiap kalimat dinilai terhadap maksud pengguna: BM25 mini, semua kata yang ditanya harus ada, nama diri harus ada, "berapa/kapan" harus berangka, judul sumber yang cocok diutamakan, dan sumber yang sesuai jenis pertanyaan diberi bobot lebih. Dari situ agent menulis:

1. jawaban singkat sesuai bentuk yang ditanya ("3 bulan", "6 Juni 1901", "Andrea Hirata", "38 provinsi"), diambil dengan pola bentuk jawaban atau GPT pembaca; untuk beberapa angka dalam satu kalimat, dipilih yang paling dekat dengan kata yang ditanya;
2. kalimat bukti terbaik dari beberapa sumber, bernomor [n];
3. daftar sumber, catatan "bukan dasar hukum" untuk internet, dan jejak alat (`otomatis ✗ → wiki-subjek ✗ → wiki-artikel "Soekarno" ✓`).

Sapaan, pemahaman maksud, hitungan, dan tanya balik tetap memakai jawaban alatnya sendiri. Jika tidak ada kalimat yang cukup cocok, agent mengatakan "belum menemukan jawaban yang benar-benar cocok", menampilkan yang paling mendekati, lalu mengajukan pertanyaan balik yang spesifik.

**Penalaran atas fakta (`src/agent/reasoner.ts`, `src/agent/relations.ts`).** Ada pertanyaan yang jawabannya berubah dari waktu ke waktu, misalnya "di mana sekarang zudan bekerja?" atau "apa jabatan X sekarang?". Informasinya juga tersebar di beberapa artikel. Untuk pertanyaan seperti ini, agent tidak mengutip satu kalimat:

1. **Relasi.** Kata "bekerja" juga terpenuhi oleh *menjabat, dijabat oleh, ditunjuk sebagai, Kepala …*. Kata "sekarang/kini/saat ini" adalah petunjuk waktu, bukan kata yang wajib ada di kalimat.
2. **Kumpulkan dulu.** Agent tidak berhenti pada hasil pertama yang cocok. Ia membaca artikel subjek, artikel lengkapnya, dan artikel lain yang menyebut nama itu (misalnya artikel lembaga tempat orang itu bekerja). Nama yang diketik huruf kecil ("zudan") dikenali karena tidak ada di kosakata indeks lokal.
3. **Fakta + waktu.** Dari setiap kalimat diambil jabatan/tempat, tahun mulai, tahun selesai, dan apakah masih berjalan ("sejak 7 Januari 2025"). Fakta hanya dipakai jika orang itu PEMEGANG jabatannya. Kalimat seperti "… mendampingi Menteri Keuangan Sri Mulyani" atau "dikenal sebagai Jokowi" tidak dihitung. Fakta yang sama dari beberapa artikel digabung dan semua sumbernya dicatat.
4. **Simpulkan menurut waktu.** Fakta yang dimulai paling akhir dan belum berakhir dianggap keadaan sekarang. Jabatan "sejak 1998" tanpa tanggal selesai dianggap sudah lewat jika ada jabatan lain yang dimulai sesudahnya. Jika tidak ada yang masih berjalan, agent menulis "yang paling akhir tercatat …".
5. **Jawaban.** Isinya kesimpulan, alasan (mengapa fakta lain tidak dipakai), dan riwayat urut waktu dengan nomor sumber.

```
Saat ini Zudan Arif Fakrulloh bekerja sebagai Kepala Badan Kepegawaian Negara (BKN) (sejak 7 Januari 2025). [1][2][3]
Alasan kesimpulan:
• Ditemukan 8 fakta "pekerjaan" tentang Zudan Arif Fakrulloh di 2 sumber.
• "Sekretaris Badan Nasional Pengelola Perbatasan" sudah berakhir (Maret 2023 – Januari 2025).
• "Direktur Jenderal Kependudukan dan Pencatatan Sipil" sudah berakhir (1 Juli 2015 – 15 Maret 2023). …
Riwayat yang ditemukan (urut waktu):
• Tahun 2008 – 2011: Kepala Bagian Penyusunan Rancangan Peraturan Perundang-Undangan Biro Hukum [3]
• … • sejak 7 Januari 2025: Kepala Badan Kepegawaian Negara (BKN) [1][2][3]  ← keadaan terkini
```

Batasannya: fakta diambil dengan pola kalimat ("sebagai …", "dijabat oleh …", "sejak …", "… hingga …"), jadi kalimat yang ditulis dengan cara lain bisa terlewat. Hasilnya juga hanya sebaru artikel Wikipedia-nya. Relasi yang dikenal: pekerjaan, tempat tinggal, kelahiran, kematian, pasangan, pendidikan. Tambahkan relasi lain di `RELASI`.

Contoh hasil (dengan internet):

| Pertanyaan | Jawaban agent | Jalan yang ditempuh |
|---|---|---|
| berapa lama cuti tahunan? | **12 hari kerja** | dokumen aturan (penyusun biasa menjawab 6 hari, padahal itu batas penangguhan) |
| berapa lama cuti besar? → syaratnya? | **3 bulan** → bekerja paling singkat 5 tahun | aturan; elipsis ditulis ulang |
| kapan Soekarno lahir? → siapa istrinya? | **6 Juni 1901** → Ratna Sari Dewi Soekarno, dkk. | artikel subjek → artikel lengkap; subjek dibawa ke pertanyaan lanjutan |
| berapa tinggi gunung Rinjani? | **3.726 mdpl** | 5 alat; "Universitas Gunung Rinjani … 1996" ditolak karena tanpa satuan |
| berapa jumlah provinsi di Indonesia? | **38 provinsi** | Wikipedia internet |

Yang perlu diketahui:

- **Internet hanya lewat API resmi.** Bawaannya Wikipedia, ditambah DuckDuckGo Instant Answer untuk definisi/entitas; keduanya tanpa kunci. Brave Search aktif jika `BRAVE_API_KEY` diisi. Halaman hasil Google, DuckDuckGo, atau Bing tidak di-scrape.
- **Situs yang diblokir jaringan dilewati otomatis.** Di Indonesia, DuckDuckGo diblokir penyedia internet (Internet Positif). Akibatnya koneksinya gagal dengan galat seperti "certificate has expired" atau "handshake failure". Setelah satu kali gagal seperti ini, situs itu dilewati selama 15 menit, dan panel cara kerja menuliskan alasannya. Agent tetap bekerja dengan Wikipedia dan sumber lokal.
- **Kata kunci pertanyaan dikirim ke layanan tersebut.** Pakai `--tanpa-internet` jika perlu, juga untuk `bun run web`.
- **Jawaban internet bukan dasar hukum.**
- **Pencocokan berbasis kata, bukan makna.** Kalimat yang memuat semua kata kunci bisa saja menjawab hal lain. Selalu periksa sumbernya.
- **`bun run eval-chat -- --agent` menguji mode ini tanpa internet.** Hasilnya 74/74. `--internet` mengizinkan internet, tetapi hasilnya tidak deterministik.

### RAG dengan GPT pembaca (lebih mengerti konteks)

RAG (*Retrieval-Augmented Generation*) berarti: **ambil** teks yang relevan dulu, lalu **baca** teks itu untuk menjawab. Di proyek ini bagian "generation"-nya adalah **GPT pembaca**. GPT pembaca tidak menulis bebas; ia **menunjuk rentang jawaban** di konteks, seperti BERT-QA/SQuAD 2.0. Karena jawabannya selalu disalin dari konteks, ia tidak bisa mengarang angka atau nama.

Cara memakai: mode **RAG (GPT pembaca)** di web, `bun run chat -- --rag`, atau ketik `:rag` di terminal.

```
pesan + riwayat + memori
  1. pahami konteks percakapan  "syaratnya?" setelah "berapa lama cuti besar?" -> "syarat cuti besar?"
                                dicari DUA kali (asli vs berkonteks), dipakai yang hasilnya lebih kuat
  2. ambil konteks              hasil cari + pasal yang dirujuk ("sebagaimana dimaksud dalam Pasal 5")
                                + lanjutan pasal + catatan memori Anda yang relevan
  3. GPT pembaca                <|user|> pertanyaan <|bot|> konteks  ->  peluang AWAL & AKHIR jawaban per token
  4. verifikasi                 keyakinan ≥ ambang, bentuk cocok ("berapa" -> angka), kalimat asal memuat kata topik
                                lolos -> "Jawaban singkat: …" + penjelasan & dasar hukum dari penyusun aturan
                                gagal -> penyusun aturan seperti biasa (tidak pernah lebih buruk)
```

**Kepala pembaca (matematikanya).** Keluaran blok terakhir `h_t` (D angka per token) dikalikan `W_span` (D×2):

- `z_awal(t) = h_t · W_span[:,0] + b_0` dan `z_akhir(t) = h_t · W_span[:,1] + b_1`
- `p_awal = softmax(z_awal)` atas posisi `<|bot|>` dan sesudahnya; posisi `<|bot|>` berarti "tidak ada jawaban"
- loss = `−log p_awal(i*) − log p_akhir(j*)`, gradien `∂L/∂z_t = p_t − 1[t = target]`
- rentang terbaik = `argmax p_awal(i)·p_akhir(j)` dengan `i ≤ j ≤ i+30`
- keyakinan = `p(rentang) / (p(rentang) + p(tidak ada))`

Turunannya diuji dengan gradient check di `bun run test-gpt`.

**Data latih tanpa label manual** (`bun run make-reader-data`, `src/gpt/reader-data.ts`). Pertanyaan dibuat dari pola kalimat Wikipedia:

| Pola kalimat | Pertanyaan buatan | Rentang jawaban |
|---|---|---|
| "Virus adalah agen infeksi …" | "apa itu virus?" | "agen infeksi …" |
| "… selama 12 hari kerja" | "berapa lama …?" | "12 hari kerja" |
| "… didirikan pada tahun 1663" | "kapan … didirikan?" | "1663" |
| "… ditulis oleh Andrea Hirata" | "siapa yang … ditulis?" | "Andrea Hirata" |
| "… di Yerusalem" | "di mana …?" | "Yerusalem" |

Konteks berisi kalimat sumber, kalimat tetangga, dan kalimat pengecoh dari artikel lain. ±30% contoh sengaja **tanpa jawaban**, supaya model belajar berkata "tidak ada". Val dibagi **per artikel**, jadi angka val adalah artikel yang belum pernah dilihat. Dokumen aturan tidak ikut dilatih secara bawaan; pakai `--dengan-aturan` jika mau.

**Pelatihan** (`bun run train-reader`) memulai dari GPT pretrain "wiki", lalu konteks diperpanjang 128 → 256 token. Ini bisa karena positional encoding sinusoidal *dihitung*, bukan dipelajari. Di akhir, ambang keyakinan dikalibrasi di data val.

**Hasil saat ini** (`bun run eval-reader`, 1000 contoh dari artikel val yang belum pernah dilihat):

| Pola | EM | F1 |
|---|---|---|
| waktu | 73,1% | 77,6% |
| siapa | 73,5% | 78,9% |
| tempat | 69,6% | 74,4% |
| besaran | 63,0% | 76,6% |
| definisi | 29,4% | 72,7% |
| **total** | **65,3%** | **76,0%** |

- Contoh tanpa jawaban yang ditolak dengan benar: 60%. Kecepatan ±45 ms per contoh.
- Riwayat latih: 622 langkah dari "wiki" (F1 59%), lalu 1800 langkah lanjutan (F1 76%). Waktunya ±47 menit di CPU.

**Pada dokumen aturan** (`bun run eval -- --rag`), pembaca belum banyak membantu. Ia hanya dilatih dari Wikipedia; bahasa pasal ("12 (dua belas) hari kerja", "ayat (1)") berbeda. Hasilnya: 1 dari 21 jawaban lolos verifikasi, dan yang lolos itu benar. Rentang yang salah ("6 (en", "3 (1) Uang hari") ditolak verifikasi. Jadi akurasi tetap 90,5%, tidak turun.

Untuk dokumen Anda sendiri, latih ulang dengan data dari dokumen aturan juga:

```
bun run make-reader-data -- --dengan-aturan
bun run train-reader -- --from reader --name reader2 --steps 1500
```

Catatan: setelah itu, skor `bun run eval` tidak lagi sepenuhnya "belum pernah dilihat".

**Konteks percakapan, dokumen, dan memori (berlaku juga tanpa RAG):**

- **Elipsis aturan.** "syaratnya?" setelah "berapa lama cuti besar?" ditulis ulang menjadi "syarat cuti besar?". Jika pengguna ganti topik ("berapa hari cuti tahunan?"), pencarian tanpa konteks yang menang.
- **Lanjutan internet.** Jawaban internet di mode agent kini menyimpan topik. "kapan dia lahir?" setelah membahas Soekarno dicari sebagai "kapan Soekarno lahir?".
- **Pasal yang dirujuk** dan lanjutan pasal ikut dibaca (mode RAG).
- **Catatan memori pribadi.** Contohnya "saya sedang sakit". Untuk pertanyaan tentang diri Anda ("berapa lama cuti yang bisa saya ambil?"), catatan itu menjadi konteks pencarian, dan dipakai jika hasilnya lebih kuat (mode RAG). Catatan yang dipakai disebutkan di jawaban.

### Memori

Memori disimpan **manual** oleh Anda dan tetap ada setelah chatbot ditutup (`data/memory.json`). Di terminal pakai `:ingat`, `:memori`, dan `:lupa`. Di web pakai panel **Memori** di bawah percakapan, atau tautan "simpan ke memori" di setiap pesan Anda.

Memori dipakai dengan dua cara:

- **Dicari seperti dokumen.** Simpan "rapat tim setiap Senin jam 9", lalu tanya "kapan rapat tim?". Jawabannya mengutip catatan itu dengan sudut pandang dibalik: "nama saya Irun" menjadi "Nama Anda Irun."
- **Nama dipakai GPT obrolan.** Jika ada catatan seperti "nama saya Irun", nama itu dimasukkan sebagai giliran perkenalan di awal prompt GPT, sehingga GPT bisa menyapa dengan nama Anda.

### Tampilan web

`bun run web` membuka chatbot di http://localhost:3000 (ganti port dengan `--port 8080`). Halaman dibagi dua:

- **Kiri: percakapan.** Setiap jawaban diberi lencana jalurnya (dokumen aturan, Wikipedia, GPT obrolan, atau tidak ditemukan). Nomor `[1]` bisa diklik untuk membuka teks lengkap sumbernya.
- **Kanan: cara kerja.** Klik sebuah jawaban untuk melihat setiap tahapnya. Setiap tahap punya kotak "Penjelasan sederhana" untuk orang awam dan perhitungan dengan angka sungguhan:
  - **Pencarian:** hitungan BM25 kandidat teratas per kata (IDF dengan angka N dan n, frekuensi f, faktor panjang, sumbangan setiap kata), lalu hitungan cakupan dan kekuatan yang menentukan keyakinan.
  - **GPT:** token prompt, vektor embedding + positional encoding + hasil penjumlahannya, simulasi attention untuk satu token (q·k, dibagi √16, softmax), peta panas attention untuk setiap lapisan dan head, perhitungan skor → suhu → e^x → peluang, serta setiap token yang dibuat beserta peluangnya.

**Ekspor untuk perbaikan.** Tombol **Salin percakapan** menyalin seluruh percakapan, termasuk jalur, keyakinan, langkah proses, dan token GPT, sebagai teks yang bisa langsung ditempel saat meminta perbaikan. **Unduh JSON** menyimpan data mentah lengkap, termasuk matriks. Di terminal, perintah `:export` mencetak teks yang sama dan menyimpannya di folder `exports/`.

Untuk uji coba, pilih **Mode** (otomatis, hanya aturan, hanya Wikipedia, atau hanya GPT) dan geser **Suhu GPT** untuk melihat pengaruhnya pada balasan. Di terminal, ringkasan yang sama tampil dengan `bun run chat -- --proses` atau perintah `:proses`.

**Halaman data & training (`/latih`).** Halaman ini menjalankan setiap langkah dengan tombol: ① ambil data (Wikipedia, JDIH, data obrolan/pembaca, upload file), ② siapkan dataset, ③ latih (model baru / lanjutkan / fine-tune, chatbot aturan, indeks Wikipedia, GPT pembaca), ④ evaluasi. Ada juga peta data (setiap folder, ukuran, dipakai langkah apa). Panel **Proses** menampilkan progress bar, grafik loss, log berjalan, dan tombol berhenti aman. Perintah yang boleh dijalankan dibatasi daftar putih di `src/web/jobs.ts`. Server hanya mendengar `127.0.0.1` kecuali diberi `--host`, dan menjalankan proses atau upload hanya diterima dari komputer yang menjalankan server. Proses yang dimulai dari web tetap berjalan walau server web dimatikan.

Logika percakapan ada di `src/chat-engine.ts` dan dipakai bersama oleh terminal maupun web.

Setiap pesan diproses berurutan:

| Urutan | Jenis pesan | Ditangani oleh |
|---|---|---|
| 0 | Perkenalan dan pertanyaan tentang diri Anda ("namaku Irun", "siapa aku?", "tau aku?") | Pemahaman maksud dengan pola tetap, nama dari memori atau dari percakapan ini |
| 0 | "lebih detail", "lanjutkan" | Teks lengkap sumber jawaban sebelumnya |
| 1 | Sapaan, basa-basi, perkenalan ("halo", "terima kasih", "nama saya Irun") | GPT percakapan, berlabel "bukan dasar hukum" |
| 2 | Pertanyaan aturan | Kutipan pasal asli + dasar hukum |
| 3 | Pertanyaan tentang diri Anda ("siapa nama saya?") | Memori yang Anda simpan |
| 4 | Pesan untuk bot ("siapa penciptamu?", "bagaimana kabarmu?") | GPT percakapan |
| 5 | Pertanyaan fakta umum ("apa itu virus?", "Oxalaia apa itu?") | Kutipan Wikipedia + judul artikel, hanya jika yakin |
| 6 | Topik lain yang ada di memori ("kapan rapat tim?") | Memori yang Anda simpan |
| 7 | Obrolan lain yang bukan pertanyaan | GPT percakapan |
| 8 | Pertanyaan yang tidak ada sumbernya | "Maaf, saya tidak menemukan ..." (tidak dikarang) |

Untuk pertanyaan definisi, jawaban utamanya adalah kalimat berbentuk "<subjek> adalah ...". Kalimat ini dicari di paragraf pembuka artikel, misalnya "Virus adalah agen infeksi berukuran kecil ...".

Kenapa fakta tidak dijawab GPT? Model 0,46 juta parameter tidak mampu menyimpan fakta. Sistem sungguhan pun mengambil fakta dari dokumen (retrieval), lalu memakai model bahasa untuk merangkai percakapan.

## Contoh jawaban chatbot

```
Kamu> berapa lama cuti melahirkan?
Bot> 3 bulan.
     Lamanya cuti melahirkan adalah 3 bulan. [1]

     Ketentuan terkait:
     • Pegawai perempuan yang akan melahirkan berhak atas cuti melahirkan. [1]

     Dasar hukum:
       [1] Peraturan Contoh Nomor 1 Tahun 2026 tentang Cuti Pegawai, Pasal 5 ayat (1), (2)
           samples/sample-employee-leave.txt, baris 49

Kamu> kalau cuti besar?
Bot> 3 bulan.
     Pegawai yang telah bekerja paling singkat 5 tahun ... berhak atas cuti besar paling lama 3 bulan. [1]
```

## Cara kerja chatbot

```
pertanyaan -> jenis (jumlah / ya-tidak / definisi / syarat / prosedur / sanksi / waktu)
           -> perluasan kata (kata dasar, singkatan, sinonim, tetangga word2vec, konteks chat)
           -> peringkat BM25 (+ peringkat makna) -> gabung dengan RRF
           -> pecah pasal teratas jadi kalimat -> pilih kalimat sesuai jenis
           -> kesimpulan + rincian + dasar hukum + catatan perubahan
```

**BM25** menjumlahkan kecocokan setiap kata q di pasal D:

```
idf(q)     = ln( 1 + (N - n_q + 0.5) / (n_q + 0.5) )
skor(D, Q) = Σ_q w_q · idf(q) · f(q,D)·(k1+1) / ( f(q,D) + k1·(1 - b + b·|D|/avgdl) )
```

**Keyakinan** memakai dua ukuran pada pasal teratas:

| Ukuran | Rumus | Gunanya |
|---|---|---|
| Cakupan | Σ idf kata yang cocok / Σ idf semua kata | Seberapa lengkap pertanyaan terjawab |
| Kekuatan | Σ idf kata yang cocok / ln(1+N) | Menolak jika yang cocok hanya kata umum |

Jawaban berkeyakinan tinggi jika cakupan minimal 60%, dan sedang jika minimal 35%. Keduanya juga mensyaratkan kekuatan minimal 0.9. Ambangnya ada di `src/answer/compose.ts`.

| Set uji chatbot | Pasal benar di peringkat 1 | Di 5 teratas | Jawaban benar | Penolakan benar |
|---|---|---|---|---|
| 5 dokumen contoh, 21 soal | 85.7% | 95.2% | 90.5% | 3 dari 3 |
| 4 PDF peraturan asli, 11 soal | 54.5% | 72.7% | 72.7% | 3 dari 3 |

## GPT dari nol

Arsitekturnya mengikuti diagram decoder-only transformer:

```
token ──► Embedding ──(+)── Positional Encoding (sinusoidal)
                       │
     ┌─────────────────▼──────────────────┐
 N×  │ Causal Multi-Head Attention          │
     │ Add & Layer Norm    h = LN(x + A(x)) │   post-LN (bawaan, sesuai diagram)
     │ Feed Forward                         │
     │ Add & Layer Norm    y = LN(h + F(h)) │
     └─────────────────┬──────────────────┘
                  Linear ──► Softmax ──► peluang token berikutnya
```

Opsi `--pre-ln` memakai susunan GPT-2: `h = x + A(LN(x))`, `y = h + F(LN(h))`, ditambah Layer Norm akhir.

**Rumus inti** (semua turunan ditulis tangan di `src/gpt/layers.ts` dan diverifikasi `bun run test-gpt`):

```
Embedding    x_t = √D · E[token_t] + PE_t
Posisi       PE(pos, 2i) = sin(pos / 10000^(2i/D)),  PE(pos, 2i+1) = cos(...)
Attention    A = softmax(Q·Kᵀ/√d_h + M) · V,   M_ij = −∞ jika j > i (causal)
Layer Norm   y = γ · (x − μ)/√(σ² + ε) + β
Feed Forward F(x) = GELU(x·W1 + b1)·W2 + b2
Keluaran     p = softmax(x·W_out + b),   L = −(1/T) Σ log p[target]
Gradien      ∂L/∂logits = p − onehot(target)
Adam         m ← β1·m + (1−β1)·g,  v ← β2·v + (1−β2)·g²,  θ ← θ − lr·m̂/(√v̂ + ε)
Dropout      y = x ⊙ m/(1−p),  m_i ∈ {0, 1} acak       (mencegah menghafal)
Sampling     p_i ∝ exp(z_i / suhu), lalu repetition penalty, top-k, dan top-p
```

**Opsi `train-gpt`** (bawaan dalam kurung):

| Opsi | Arti |
|---|---|
| `--data` | file `.txt`, folder dokumen, atau `.jsonl` percakapan; boleh beberapa dipisah koma |
| `--size` (small) | preset: small = dim 64, 4 lapisan, konteks 128, vocab ≤ 2000; medium = dim 96, vocab ≤ 3000 |
| `--layers --dim --heads --context --ff --vocab` | menimpa preset |
| `--minutes` (45, chat 10) | anggaran waktu; jumlah langkah dihitung dari kecepatan nyata |
| `--steps` | jumlah langkah tetap, menggantikan `--minutes` |
| `--batch` (16), `--lr` (0.003, fine-tune 0.001) | ukuran batch dan learning rate |
| `--patience` (5) | early stopping setelah N evaluasi tanpa perbaikan |
| `--dropout` (0.1), `--weight-decay` (0.1) | regularisasi |
| `--from <nama>` | fine-tune dari model lain (bobot + tokenizer) |
| `--resume` | lanjutkan model `--name` |
| `--keep-case`, `--pre-ln`, `--workers`, `--seed` | lain-lain |

**Opsi `generate`**: `--length` (200 token), `--temperature` (0.8), `--top-k` (40), `--top-p` (0.9), `--repeat-penalty` (1.15), `--stop-newline`, `--seed`.

## GPT yang paham konteks percakapan

Model obrolan lama (`chat`) sulit memakai informasi dari giliran sebelumnya, karena empat hal:

1. **Data latihnya.** Kalimat templat satu niat per giliran: balasannya tidak pernah bergantung pada riwayat.
2. **Modelnya kecil.** ±0,46 juta parameter, konteks 128 token, kosakata 2.000. Kata Indonesia terpecah jadi banyak token, sehingga riwayat cepat habis.
3. **Riwayat yang dikirim sedikit.** Hanya 2 giliran obrolan terakhir.
4. **Tidak ada KV-cache.** Konteks panjang jadi lambat.

Yang diubah:

| Bagian | Perubahan | File |
|---|---|---|
| Data | percakapan **berkonteks**: ingat fakta, pilih satu dari beberapa fakta, kata ganti ("dia suka makan apa?"), koreksi ("eh salah, maksudku Bandung"), topik lanjutan, ringkas, suasana, "belum tahu", kartu konteks | `src/gpt/chat-context-data.ts` |
| Nilai uji terpisah | nama, kota, pekerjaan, hobi, makanan, dan nama hewan dibagi latih/uji (setiap nilai ke-5 = uji), sehingga eval memakai nilai yang tidak pernah dilihat | `daftarSlot()` |
| Kartu konteks | `<|konteks|> nama: Sari; tinggal: Medan; pekerjaan: guru<|end|>` di awal prompt; diisi dari memori dan pesan-pesan sesi; tidak pernah dipotong | `src/gpt/context-card.ts`, token baru di `src/gpt/bpe.ts` |
| Riwayat | hingga 6 giliran terakhir dari semua jalur (jawaban panjang diringkas), sebanyak muat di konteks | `src/chat-engine.ts` |
| Model | preset `besar`: dim 128, 6 lapisan, 4 head, FF 512, konteks 256, kosakata 4.000, pre-LN (±2,2 juta parameter) | `src/commands/train-gpt.ts` |
| Kecepatan | KV-cache: token baru hanya menghitung 1 baris; Key/Value token lama dipakai ulang. Hasilnya identik dengan forward penuh (diuji di `test-gpt`) | `GPT.forwardTambah` |
| Pengamatan | panel GPT di web: kartu konteks + "perhatian ke konteks" (persen attention ke kartu, riwayat, pesan sekarang, balasan sendiri; per lapisan; token lama yang paling diperhatikan) | `src/gpt/inspect.ts` |

Cara melatih (CPU, bisa dicicil; rincian di [PANDUAN.md](PANDUAN.md) langkah 1–3):

```
bun run fetch-data -- --target-mb 60                                                    # 1. lebih banyak Wikipedia
bun run siapkan-data -- --sumber wikipedia --nama wiki-v1 --huruf-asli                  # 2. tokenizer + token
bun run train-gpt -- --dataset wiki-v1 --name wiki2 --size besar --minutes 60           # 3. pretraining
bun run train-gpt -- --dataset wiki-v1 --name wiki2 --resume --minutes 60               #    ulangi beberapa kali
bun run make-chat-data                                                                  # 1. obrolan biasa + berkonteks
bun run siapkan-data -- --sumber obrolan --nama obrolan-v1 --tokenizer-dari wiki2       # 2. tokenizer SAMA dengan wiki2
bun run train-gpt -- --dataset obrolan-v1 --from wiki2 --name chat2 --steps 3000 --eval-every 200   # 3. fine-tune
bun run eval-konteks -- --name chat2 --banding chat                                     # 4. bandingkan dengan model lama
```

**Cara mengukur (`bun run eval-konteks`).** Percakapan uji memakai nilai slot yang tidak pernah dilatih. Balasan dianggap benar jika memuat nilai yang diharapkan, misalnya kota yang disebut tiga giliran sebelumnya. Ada juga kolom **ablasi**: riwayat dan kartu dihapus, hanya pertanyaan terakhir. Jika akurasi tanpa konteks tetap tinggi, model sekadar menebak. Jika turun drastis, model benar-benar membaca konteks.

**Hasil sekarang** (300 percakapan uji, CPU 16 thread):

| Model | Pretraining | Fine-tune | Benar (dengan konteks / ablasi) | Perplexity jawaban bot |
|---|---|---|---|---|
| `chat` (lama, small) | `wiki`, 3,1 juta token | 1.620 percakapan | 1% / 1% | 274 |
| `chat2` (besar) | `wiki2`, **hanya 300 langkah** (loss val 5,40) | 9.520 percakapan, 800 langkah (loss val 0,26) | 2% / 1% | 3,98 |

Artinya:

- **Bentuk jawaban sudah benar.** `chat2` menjawab "namaku siapa?" dengan "Tentu, nama Anda …" dan "aku tinggal di mana?" dengan "Anda tinggal di …"; `chat` masih membalas dengan sapaan atau penjelasan fitur.
- **Nilainya belum disalin dari konteks.** Isian diambil dari nilai yang dihafal saat latih ("Anda tinggal di Pekanbaru", padahal pengguna menyebut Blitar). Ablasi hampir sama, jadi model belum benar-benar membaca riwayat.
- **Penyebab utamanya pretraining yang terlalu pendek.** `wiki2` baru 300 langkah (±0,3 juta token). Kemampuan "menyalin token dari belakang" (induction head) biasanya baru muncul setelah jutaan token. Langkah berikutnya: lanjutkan `wiki2` dengan `--resume` beberapa jam, fine-tune ulang `chat2`, lalu ukur lagi dengan `eval-konteks`.
- **Chatbot tidak dirugikan.** `eval-chat` tetap 74/74 di mode normal, agent, dan RAG. Fakta tentang pengguna (nama, kota, dll.) tetap dijawab oleh jalur memori/maksud, bukan oleh GPT.

## Agar hasil GPT bagus

**Data harus jauh lebih banyak dari parameter.** Aturan praktisnya sekitar 20 token latih per parameter. Model bawaan punya ±0,46 juta parameter, jadi butuh jutaan token. `data/corpus.txt` hanya ±1.500 token, sehingga model pasti menghafal.

**Baca dua angka loss.** Loss train mengukur seberapa baik model menebak teks yang ia pelajari. Loss val mengukur teks yang tidak pernah ia lihat.

| Pola | Artinya | Tindakan |
|---|---|---|
| Keduanya turun | Model sedang belajar | Biarkan berjalan atau tambah waktu |
| Train turun, val naik | Model menghafal (overfitting) | Tambah data, perkecil model, atau naikkan dropout |
| Keduanya macet tinggi | Model terlalu kecil atau lr salah | Perbesar model atau ubah `--lr` |

Perplexity = e^(loss val), bisa dibaca sebagai "model ragu di antara sekian token". Early stopping otomatis berhenti saat loss val tidak membaik, dan checkpoint yang disimpan selalu yang loss val-nya terbaik.

**Dua tahap seperti ChatGPT.** Pretraining di Wikipedia mengajari bahasa. Fine-tuning dengan data obrolan (`data/mentah/obrolan/chat-id.jsonl`) mengajari format percakapan: setelah `<|bot|>` model membalas, lalu berhenti dengan `<|end|>`. Saat fine-tuning, loss hanya dihitung pada token jawaban bot (masking), sehingga model belajar membalas, bukan meniru pertanyaan.

**Menambah kemampuan obrolan.** Tambahkan niat dan contoh kalimat di `src/gpt/chat-data.ts`, jalankan `bun run make-chat-data` dan `bun run siapkan-data -- --sumber obrolan --nama obrolan-v2 --tokenizer-dari wiki`, lalu fine-tune ulang dari `wiki`.

**Training paralel.** Batch dibagi ke beberapa worker thread. Bobot dibagi lewat `SharedArrayBuffer`, dan gradien semua worker dijumlahkan sebelum satu update Adam. Hasilnya identik dengan satu thread, hanya lebih cepat.

**Hasil uji GPT**

Diukur di CPU 16 thread, ukuran small (0,46 juta parameter).

| Tahap | Data | Waktu | Loss val awal → akhir | Perplexity akhir |
|---|---|---|---|---|
| Korpus kecil (lama) | 1.200 token | 12 menit | 5,4 → naik ke 12 (menghafal) | 227 (checkpoint terbaik) |
| Pretraining Wikipedia | 3,1 juta token | 49 menit | 7,63 → 4,31 | 74 |
| Fine-tuning percakapan | 1.620 percakapan | 10 menit | 5,97 → 0,05 | 1,05 |

Contoh teks pretraining: *"Pada tahun 1995, ia menjalankan ..., setelah itu ..., dan kemudian menentang penelitian ..."*. Pola kalimat bahasa Indonesia sudah terbentuk, tetapi isinya belum bermakna.

Contoh percakapan setelah fine-tuning:

```
Kamu> halo                 GPT> Hai, selamat datang. Silakan sampaikan pertanyaan Anda.
Kamu> apa kabar?           GPT> Kabar saya baik, terima kasih sudah bertanya. Bagaimana dengan Anda?
Kamu> kamu bisa apa?       GPT> Saya bisa membantu mencari dan menjelaskan isi aturan, ...
Kamu> selamat malam pak    GPT> Selamat malam! Ada yang bisa saya bantu?
Kamu> ceritakan lelucon    GPT> Kenapa komputer jarang kelaparan? Karena sudah punya banyak byte.
```

Perplexity 1,05 pada percakapan berarti model hampir hafal pola dataset. Itu memang tujuannya untuk sapaan, tetapi model tidak bisa membalas topik yang jauh dari dataset. Obrolan dijawab dengan suhu 0,4 dan 2 giliran riwayat obrolan terakhir. Di `bun run chat-gpt`, riwayat diatur dengan `--history N` (bawaan 2).

## Batasan

- **Jawaban aturan tidak memakai GPT.** GPT kecil yang dilatih di CPU menghasilkan teks bergaya peraturan, tetapi isinya sering mengarang pasal dan angka. Karena itu `:gpt` selalu diberi label "bukan dasar hukum".
- **GPT butuh data dan waktu.** Ukuran bawaan sekitar 0,46 juta parameter, dilatih ±45 menit di CPU. Hasilnya runtut per frasa, tetapi isinya sering tidak masuk akal. Model sebesar ChatGPT punya ratusan miliar parameter.
- **Obrolan GPT terbatas pada pola yang dilatih.** Model membalas sapaan dan basa-basi yang mirip dataset percakapan. Pertanyaan bebas di luar itu bisa dibalas dengan kalimat yang tidak nyambung.
- **Teks Wikipedia berlisensi CC BY-SA 4.0.** Daftar artikel sumber ada di `data/mentah/wikipedia/sumber.txt`.
- **PDF hasil scan tanpa OCR** dilewati, dan kualitas OCR menentukan akurasi chatbot maupun GPT.
- **Word2vec chatbot butuh data besar.** Di bawah 200 ribu token, word2vec otomatis dilewati karena justru menurunkan akurasi.
- **Ambang keyakinan chatbot** disetel dari soal uji yang sedikit. Tulis 30 sampai 50 soal dari dokumen Anda sendiri, lalu setel ulang.

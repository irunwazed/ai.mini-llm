# LLM dari Nol + Chatbot Aturan Pemerintah + GPT Mini

Proyek belajar tiga jalur, semuanya **Bun + TypeScript tanpa library ML**:

1. **`lessons/`** belajar model bahasa dari nol secara matematika: bigram, n-gram, BPE, attention, GPT mini.
2. **Chatbot aturan** menjawab pertanyaan dari ribuan dokumen aturan (PDF, DOCX, TXT, MD). Jawabannya berupa kesimpulan, rincian, dan **dasar hukum** (peraturan, pasal, ayat, file, halaman). Setiap fakta diambil dari kalimat asli dokumen, jadi chatbot tidak mengarang.
3. **GPT** (decoder-only transformer) yang dilatih dari nol dan **menghasilkan teks** token demi token, mengikuti diagram di bawah.

Nama file dan folder memakai bahasa Inggris. Teks yang dipelajari model dan percakapan chatbot berbahasa Indonesia.

## Struktur folder

```
llm/
├── documents/          ← TARUH FILE ATURAN DI SINI (.pdf .docx .txt .md, boleh bersubfolder)
│   └── samples/        5 dokumen contoh FIKTIF untuk uji coba (boleh dihapus)
├── model/              ← hasil training chatbot (dibuat oleh `bun run train`)
├── gpt-models/         ← hasil training GPT, satu folder per nama model (dibuat oleh `bun run train-gpt`)
├── data/
│   ├── synonyms.txt    kamus padanan kata untuk chatbot (tanpa training ulang)
│   └── corpus.txt      teks kecil untuk pelajaran
├── evaluation/
│   └── questions.jsonl soal uji akurasi chatbot
├── lessons/            01_bigram, 02_ngram, 03_bpe, 04_attention, 05_mini_gpt
└── src/
    ├── readers/        membaca PDF, DOCX, TXT/MD (dengan cache)
    ├── text/           pembersih header/OCR, tokenisasi, stemmer, sinonim, pemotong pasal
    ├── training/       metadata & perubahan aturan, glosarium & singkatan, word2vec
    ├── search/         BM25 + pencarian hibrida
    ├── answer/         jenis pertanyaan, pemecah kalimat, penyusun jawaban
    ├── gpt/            GPT dari nol: bpe, layers, model, optimizer, sampling, worker, dataset, checkpoint
    ├── model.ts        simpan / muat model chatbot
    └── commands/       train, search, chat, eval, train-gpt, generate, test-gpt
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
| Tanya satu pertanyaan | `bun run search -- "berapa hari cuti tahunan?"` |
| Chatbot interaktif | `bun run chat` |
| Chatbot mode agent (semua mode + internet) | `bun run chat -- --agent` (atau ketik `:agent` di dalam chat) |
| Mode agent tanpa internet | `bun run chat -- --agent --tanpa-internet` |
| Ukur akurasi jawaban aturan | `bun run eval` |
| Uji percakapan "nyambung" | `bun run eval-chat` |
| Uji percakapan lewat mode agent (tanpa internet) | `bun run eval-chat -- --agent` |
| Uji soal logika & matematika | `bun run eval-reasoning` |

**GPT, alur lengkap seperti ChatGPT (dalam versi mini)**

```
bun run test-gpt                                                         # 0. pastikan semua turunan benar
bun run fetch-data                                                       # 1. unduh ±10 MB Wikipedia Indonesia
bun run train-gpt -- --data data/wiki-id.txt --name wiki --keep-case     # 2. pretraining ±45 menit
bun run generate -- --name wiki "Indonesia adalah"                       #    coba hasilkan teks
bun run make-chat-data                                                   # 3. buat dataset percakapan
bun run train-gpt -- --data data/chat-id.jsonl --from wiki --name chat --steps 1500   # 4. fine-tuning chat ±10 menit
bun run chat-gpt                                                         # 5. ngobrol dengan GPT
bun run index-wiki                                                       # 6. indeks Wikipedia untuk pertanyaan fakta
bun run chat                                                             # 7. chatbot lengkap: aturan + fakta + obrolan
bun run web                                                              # 8. versi web + panel "cara kerja" -> http://localhost:3000
```

**Penting saat training:**

- **Jangan jalankan dua training dengan `--name` yang sama sekaligus.** Keduanya menulis ke folder yang sama, sehingga checkpoint saling menimpa. Keduanya juga berebut CPU, sehingga masing-masing jauh lebih lambat.
- **Cegah komputer tidur.** Jika komputer tidur di tengah training berbasis `--minutes`, pengukuran kecepatan menjadi salah dan jumlah langkahnya terlalu sedikit. Untuk fine-tuning chat, `--steps 1500` lebih aman.

| Tujuan | Perintah |
|---|---|
| Lanjutkan training | `bun run train-gpt -- --data data/wiki-id.txt --name wiki --resume --minutes 30` |
| Latih dari folder aturan | `bun run train-gpt -- --data documents --name rules` |
| Gabung beberapa sumber | `--data data/wiki-id.txt,documents` |

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

Loop ini tidak bisa menjamin 100%, karena pemeriksanya juga aturan perkiraan. Tetapi kesalahan yang terdeteksi diperbaiki otomatis, dan hasilnya terukur dengan `bun run eval-chat`: 86,3% sebelum loop, 100% (58/58 giliran) sesudahnya.

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

### Mode agent (semua mode + pencarian internet)

Pilih **Agent (semua mode + internet)** di menu Mode web, atau jalankan `bun run chat -- --agent`. Kodenya ada di `src/agent/multi-mode.ts`. Tidak ada layanan AI luar.

**Putaran 1: semua mode lokal dicoba.** Mode yang dicoba: otomatis, penalaran, dokumen aturan, Wikipedia lokal, dan GPT. GPT dilewati untuk pertanyaan fakta. Setiap hasil diberi skor 0–1:

| Hasil | Skor |
|---|---|
| pola maksud ("siapa aku?"), permintaan detail | 1,0 |
| agent penalaran (lolos cek ulang) | 0,95 |
| memori | 0,85 |
| GPT untuk sapaan/obrolan | 0,8 |
| dokumen aturan / Wikipedia / internet | keyakinan × (0,2 + 0,8 × cakupan²) |
| tanya balik (pertanyaan ambigu) | 0,55 |
| jujur "tidak tahu" | 0,3 |
| GPT untuk pertanyaan fakta | 0,2 |

Cakupan adalah bagian kata pertanyaan yang ada di jawaban. Skor jawaban pencarian dikurangi dalam tiga keadaan:

- pertanyaan "berapa" dijawab tanpa angka atau tanpa satuan yang diminta;
- pertanyaan "kapan" dijawab tanpa tanggal;
- pesannya sebenarnya obrolan ("halo", "aku bosan").

**Putaran 2+: pencarian internet.** Ini hanya terjadi jika skor terbaik di bawah 0,6 dan pesannya pertanyaan. Strateginya diulang sampai ada kalimat yang lolos cek:

1. Jika ada nama diri ("Soekarno", "Rinjani"), cari artikel subjeknya di Wikipedia, lalu baca artikel lengkapnya. Contohnya "Jakarta", yang dialihkan ke "Daerah Khusus Ibukota Jakarta".
2. Cari Wikipedia dengan semua kata kunci.
3. DuckDuckGo Instant Answer (API resmi, gratis, tanpa kunci): definisi/entitas saja — untuk pertanyaan biasa hasilnya sering kosong, dan Abstract biasanya bersumber Wikipedia, jadi ini pelengkap, bukan pengganti.
4. Opsional: Brave Search API.

Kalimat halaman dinilai dengan BM25 mini (`src/answer/web-extract.ts`), lalu dikutip apa adanya beserta tautannya. Sebuah kalimat lolos cek jika memenuhi tiga syarat:

- semua kata kunci ada (jika kata kuncinya ≤ 3), atau ≥ 70% kata kunci (jika lebih);
- nama diri ada;
- untuk "berapa/kapan", kalimat memuat angka.

**Pemilihan akhir.** Kandidat berskor tertinggi dipilih. Semua langkah (Pikir/Aksi/Hasil/Cek) dan tabel skor kandidat tampil di panel cara kerja.

Contoh hasil:

- "kapan Soekarno lahir?" → "Soekarno dilahirkan di Surabaya, pada tanggal 6 Juni 1901 …"
- "berapa jumlah provinsi di Indonesia?" → "Indonesia terdiri atas 38 provinsi …"
- "siapa penulis Laskar Pelangi?" → "… novel yang ditulis Andrea Hirata."

Yang perlu diketahui:

- **Internet hanya lewat API resmi.** Yang dipakai bawaan adalah API Wikipedia dan API DuckDuckGo Instant Answer: keduanya gratis, tanpa kunci. Halaman hasil mesin pencari (Google, DuckDuckGo, Bing) tidak di-scrape — termasuk halaman HTML DuckDuckGo, karena melanggar ketentuan layanan mereka dan diblokir pemeriksaan bot; yang dipakai hanya API resminya. Untuk web umum, buat kunci Brave Search API sendiri, lalu jalankan dengan variabel lingkungan `BRAVE_API_KEY`.
- **Kata kunci pertanyaan dikirim ke Wikipedia dan DuckDuckGo** (dan ke Brave, jika diaktifkan). Jangan menanyakan data pribadi dalam mode agent. Pakai `--tanpa-internet` jika perlu (untuk web: `bun run web -- --tanpa-internet`).
- **Jawaban internet bukan dasar hukum.** Jawaban itu selalu diberi label "kutipan dari internet".
- **Pencocokan masih berbasis kata, bukan makna.** Kalimat yang memuat semua kata kunci bisa saja menjawab hal lain. Selalu periksa tautan sumbernya.
- **Mode agent lebih lambat.** Semua mode dijalankan, ditambah ±0,5–3 detik jika mencari di internet.

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

## Agar hasil GPT bagus

**Data harus jauh lebih banyak dari parameter.** Aturan praktisnya sekitar 20 token latih per parameter. Model bawaan punya ±0,46 juta parameter, jadi butuh jutaan token. `data/corpus.txt` hanya ±1.500 token, sehingga model pasti menghafal.

**Baca dua angka loss.** Loss train mengukur seberapa baik model menebak teks yang ia pelajari. Loss val mengukur teks yang tidak pernah ia lihat.

| Pola | Artinya | Tindakan |
|---|---|---|
| Keduanya turun | Model sedang belajar | Biarkan berjalan atau tambah waktu |
| Train turun, val naik | Model menghafal (overfitting) | Tambah data, perkecil model, atau naikkan dropout |
| Keduanya macet tinggi | Model terlalu kecil atau lr salah | Perbesar model atau ubah `--lr` |

Perplexity = e^(loss val), bisa dibaca sebagai "model ragu di antara sekian token". Early stopping otomatis berhenti saat loss val tidak membaik, dan checkpoint yang disimpan selalu yang loss val-nya terbaik.

**Dua tahap seperti ChatGPT.** Pretraining di Wikipedia mengajari bahasa. Fine-tuning dengan `data/chat-id.jsonl` mengajari format percakapan: setelah `<|bot|>` model membalas, lalu berhenti dengan `<|end|>`. Saat fine-tuning, loss hanya dihitung pada token jawaban bot (masking), sehingga model belajar membalas, bukan meniru pertanyaan.

**Menambah kemampuan obrolan.** Tambahkan niat dan contoh kalimat di `src/gpt/chat-data.ts`, jalankan `bun run make-chat-data`, lalu fine-tune ulang dari `wiki`.

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
- **Teks Wikipedia berlisensi CC BY-SA 4.0.** Daftar artikel sumber ada di `data/wiki-id.sources.txt`.
- **PDF hasil scan tanpa OCR** dilewati, dan kualitas OCR menentukan akurasi chatbot maupun GPT.
- **Word2vec chatbot butuh data besar.** Di bawah 200 ribu token, word2vec otomatis dilewati karena justru menurunkan akurasi.
- **Ambang keyakinan chatbot** disetel dari soal uji yang sedikit. Tulis 30 sampai 50 soal dari dokumen Anda sendiri, lalu setel ulang.

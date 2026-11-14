/**
 * TAMPILAN WEB untuk chatbot + panel "cara kerja" (bahan belajar & uji coba).
 *
 * Pakai:  bun run web                 lalu buka http://localhost:3000  (chat)  dan  http://localhost:3000/latih  (data & training)
 *         bun run web -- --port 8080
 *         bun run web -- --host 0.0.0.0   (bisa dibuka komputer lain; menjalankan proses & upload TETAP hanya dari komputer ini)
 *
 * API:
 *   GET    /api/info          ringkasan model yang dimuat (aturan, Wikipedia, GPT, memori)
 *   POST   /api/model/pakai   { jenis: "obrolan"|"pembaca", nama|null } -> ganti GPT yang dipakai chat (null = terbaru;
 *                              berlaku untuk seluruh server; hanya dari komputer ini)
 *   POST   /api/chat          { pesan, riwayat?, topik?, mode?, suhu? } -> Respons (jawaban + proses + jejak GPT)
 *   GET    /api/memori        daftar catatan memori
 *   POST   /api/memori        { teks }  -> simpan catatan baru (manual oleh pengguna)
 *   DELETE /api/memori/:id    hapus satu catatan      DELETE /api/memori   hapus semua
 *   GET    /api/progres       progres training/unduhan yang sedang berjalan (bun run status versi web)
 *   POST   /api/progres/hentikan { jenis, nama } -> minta berhenti dengan aman
 *   POST   /api/export        { entri[], mode, suhu } -> teks ekspor percakapan (markdown)
 * Halaman /latih (langkah ambil data -> siapkan -> latih -> evaluasi):
 *   GET  /api/data             peta data: setiap lokasi, ukuran, jumlah file, terakhir diubah
 *   GET  /api/model            model GPT + riwayat dataset + sidik tokenizer
 *   GET  /api/dataset          dataset siap latih (data/siap/)
 *   GET  /api/perintah         daftar putih perintah + opsi yang boleh
 *   GET  /api/job              proses yang dijalankan dari web        POST /api/job { perintah, opsi }
 *   GET  /api/job/:id/log?dari=N   log berjalan                       POST /api/job/:id/hentikan
 *   POST /api/upload           multipart: tujuan=aturan|teks, berkas=…  (hanya dari komputer ini)
 *   GET  /api/drive/status     status login Google Drive (tanpa token)   GET /api/drive/login -> ke halaman login Google
 *   GET  /drive/login          isi POPUP login (persiapan client / ke Google)   GET /drive/selesai  hasil login di popup
 *   GET  /api/drive/isi?path=  isi folder Drive vs lokal (penjelajah)   POST /api/drive/klien  simpan JSON client OAuth
 *   POST /api/drive/logout     cabut akses Drive.  Google kembali ke "/?code=…&state=…" (redirect loopback)
 *   GET  /api/server           alamat server training yang diatur (bun run server -- --atur …)
 *
 * Server tidak menyimpan riwayat percakapan: halaman mengirim riwayat & topik terakhir
 * di setiap pesan. Memori disimpan di server (data/memory.json).
 */
import { join } from "node:path";
import { eksporMarkdown, MesinChat, type EntriEkspor, type GiliranRiwayat, type Mode, type Topik } from "../chat-engine.js";
import { jalankanModeAgent } from "../agent/multi-mode.js";
import { bacaSemuaProgres, mintaBerhenti, type JenisProses } from "../gpt/progress.js";
import { existsSync, mkdirSync, readdirSync, readFileSync } from "node:fs";
import { basename, extname } from "node:path";
import { daftarDataset, infoLokasi, sidikTokenizer } from "../gpt/prepared.js";
import { LOKASI, lokasiBaca } from "../paths.js";
import { bacaLog, daftarJob, hentikanJob, mulaiJob, PERINTAH } from "../web/jobs.js";
import type { MetaGPT } from "../gpt/checkpoint.js";
import { adaLoginTertunda, buatUrlLogin, logout as logoutDrive, PETUNJUK_KLIEN, selesaikanLogin, simpanKlien, statusLogin } from "../cloud/google-auth.js";
import { jelajahDrive, pathJelajahSah } from "../cloud/browse.js";
import { halamanPersiapan, halamanSelesai } from "../web/drive-pages.js";
import { argumen } from "./args.js";

const { folderModel, k, opsi } = argumen();
const mesin = new MesinChat({ folderModel, folderWiki: opsi.wiki, chatGpt: opsi["chat-gpt"], k, fileMemori: opsi.memori });
const port = Number(opsi.port ?? 3000);
const HALAMAN = join(import.meta.dir, "..", "web", "index.html");
const HALAMAN_LATIH = join(import.meta.dir, "..", "web", "latih.html");
const html = (p: string) => new Response(Bun.file(p), { headers: { "content-type": "text/html; charset=utf-8" } });

/** Peta data untuk halaman /latih: setiap lokasi, isinya, dibuat oleh langkah apa, dipakai oleh langkah apa. */
function petaData() {
  const baris = [
    { langkah: 1, nama: "Dokumen aturan", lokasi: LOKASI.dokumenAturan, isi: "PDF/DOCX/TXT/MD peraturan", dari: "unduh JDIH, upload, salin manual", dipakai: "latih chatbot aturan; siapkan-data --sumber aturan" },
    { langkah: 1, nama: "Wikipedia", lokasi: lokasiBaca(LOKASI.wikiTeks), isi: "teks artikel Wikipedia Indonesia", dari: "unduh Wikipedia", dipakai: "indeks Wikipedia; siapkan-data --sumber wikipedia; data GPT pembaca" },
    { langkah: 1, nama: "Teks sendiri", lokasi: LOKASI.teksSendiri, isi: "teks bahasa Indonesia milik Anda (.txt)", dari: "upload, salin manual", dipakai: "siapkan-data --sumber teks" },
    { langkah: 1, nama: "Data obrolan", lokasi: lokasiBaca(LOKASI.obrolan), isi: "percakapan latih (biasa + berkonteks)", dari: "buat data obrolan", dipakai: "siapkan-data --sumber obrolan" },
    { langkah: 1, nama: "Data GPT pembaca", lokasi: lokasiBaca(LOKASI.pembaca), isi: "soal + rentang jawaban otomatis", dari: "buat data pembaca", dipakai: "latih GPT pembaca" },
    { langkah: 2, nama: "Dataset siap latih", lokasi: LOKASI.siap, isi: "tokenizer + token per dataset", dari: "siapkan data", dipakai: "latih GPT (--dataset)" },
    { langkah: 3, nama: "Model GPT", lokasi: LOKASI.modelGPT, isi: "bobot + tokenizer + riwayat loss per model", dari: "latih GPT / GPT pembaca", dipakai: "chat, evaluasi" },
    { langkah: 3, nama: "Indeks aturan", lokasi: LOKASI.modelAturan, isi: "indeks BM25 + metadata pasal", dari: "latih chatbot aturan", dipakai: "chat, evaluasi" },
    { langkah: 3, nama: "Indeks Wikipedia", lokasi: LOKASI.modelWiki, isi: "indeks BM25 Wikipedia", dari: "indeks Wikipedia", dipakai: "chat (pengetahuan umum)" },
    { langkah: 4, nama: "Soal evaluasi", lokasi: "evaluation", isi: "questions/conversations/reasoning.jsonl", dari: "ditulis manual", dipakai: "evaluasi" },
    { langkah: 0, nama: "Progres & log", lokasi: LOKASI.progres, isi: "progres + log setiap proses", dari: "semua langkah", dipakai: "panel proses" },
  ];
  return baris.map((b) => ({ ...b, ...infoLokasi(b.lokasi), ada: existsSync(b.lokasi), lama: b.lokasi.startsWith("data/") && !b.lokasi.startsWith("data/mentah") && !b.lokasi.startsWith("data/siap") }));
}

function daftarModel() {
  if (!existsSync(LOKASI.modelGPT)) return [];
  return readdirSync(LOKASI.modelGPT).filter((n) => existsSync(join(LOKASI.modelGPT, n, "gpt.json"))).map((n) => {
    const m = JSON.parse(readFileSync(join(LOKASI.modelGPT, n, "gpt.json"), "utf8")) as MetaGPT;
    return {
      nama: n, mode: m.mode ?? "teks", dasar: m.dasar ?? null, langkah: m.langkah, lossVal: m.lossValTerbaik, waktu: m.waktu,
      config: { dim: m.config.dim, lapisan: m.config.layers, konteks: m.config.context, kosakata: m.config.vocab }, hurufKecil: m.hurufKecil,
      sidik: sidikTokenizer(m.bpe), dataset: m.dataset ?? [{ nama: m.data, mulai: 1, selesai: m.langkah, waktu: m.waktu }],
    };
  }).sort((a, b) => b.waktu.localeCompare(a.waktu));
}

const EKSTENSI: Record<string, string[]> = { aturan: [".pdf", ".docx", ".txt", ".md"], teks: [".txt"] };
const galat = (pesan: string, status = 400) => Response.json({ error: pesan }, { status });

/** Menjalankan proses & upload hanya boleh dari komputer ini. */
const lokal = (req: Request): boolean => { const ip: string = server.requestIP(req)?.address ?? ""; return ip === "127.0.0.1" || ip === "::1" || ip.endsWith(":127.0.0.1"); };
const tolak = (): Response => galat("Hanya boleh dari komputer yang menjalankan server.", 403);

const portServer = (): number => server.port ?? port;
const server = Bun.serve({
  port,
  hostname: opsi.host ?? "127.0.0.1",
  maxRequestBodySize: 200 * 1024 * 1024,
  routes: {
    "/": async (req) => {
      // kembalian login Google Drive (redirect_uri = http://127.0.0.1:<port>/)
      const u = new URL(req.url), state = u.searchParams.get("state");
      if (adaLoginTertunda(state) && lokal(req)) {
        const err = u.searchParams.get("error"), code = u.searchParams.get("code");
        let pesan = "berhasil";
        try { if (err || !code) throw new Error(err ?? "tidak ada kode"); await selesaikanLogin(state!, code); }
        catch (e) { pesan = "gagal: " + (e as Error).message; }
        return Response.redirect(`/drive/selesai?hasil=${encodeURIComponent(pesan)}`, 302);
      }
      return html(HALAMAN);
    },
    // ---- jendela popup login Google Drive
    "/drive/login": (req): Response => {
      if (!lokal(req)) return tolak();
      if (!statusLogin().klien) return new Response(halamanPersiapan(), { headers: { "content-type": "text/html; charset=utf-8" } });
      try { return Response.redirect(buatUrlLogin(`http://127.0.0.1:${portServer()}/`).url, 302); }
      catch (e) { return new Response(halamanSelesai("gagal: " + (e as Error).message), { headers: { "content-type": "text/html; charset=utf-8" } }); }
    },
    "/drive/selesai": (req) => new Response(halamanSelesai(new URL(req.url).searchParams.get("hasil")?.slice(0, 400) ?? "gagal: tidak ada hasil"), { headers: { "content-type": "text/html; charset=utf-8" } }),
    "/api/drive/klien": {
      POST: async (req) => {
        if (!lokal(req)) return tolak();
        const r = simpanKlien(await req.text(), new URL(req.url).searchParams.get("ganti") === "1");
        return "galat" in r ? galat(r.galat) : Response.json(r);
      },
    },
    "/api/drive/isi": async (req) => {
      if (!lokal(req)) return tolak();
      const path = (new URL(req.url).searchParams.get("path") ?? "").replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
      if (!pathJelajahSah(path)) return galat("Folder tidak boleh dijelajah.");
      try { return Response.json(await jelajahDrive(path)); }
      catch (e) { return galat((e as Error).message, 502); }
    },
    "/api/gpu": async () => {
      const { gpuTersedia } = await import("../gpu/cuda.js");
      return Response.json({ ada: gpuTersedia() });
    },
    "/api/drive/status": () => Response.json({ ...statusLogin(), petunjuk: statusLogin().klien ? null : PETUNJUK_KLIEN }),
    "/api/drive/login": (req): Response => {
      if (!lokal(req)) return tolak();
      try { return Response.redirect(buatUrlLogin(`http://127.0.0.1:${portServer()}/`).url, 302); }
      catch (e) { return galat((e as Error).message); }
    },
    "/api/drive/logout": { POST: async (req) => (lokal(req) ? Response.json({ ok: await logoutDrive() }) : tolak()) },
    "/api/server": (req) => {
      const f = join(LOKASI.rahasia, "server.json");
      if (!existsSync(f)) return Response.json({ ada: false });
      const sv = JSON.parse(readFileSync(f, "utf8")) as { tujuan: string; port: number };
      return Response.json(lokal(req) ? { ada: true, tujuan: sv.tujuan, port: sv.port } : { ada: true });
    },
    "/latih": () => html(HALAMAN_LATIH),
    "/api/data": () => Response.json(petaData()),
    "/api/model": () => Response.json(daftarModel()),
    "/api/model/pakai": {
      POST: async (req) => {
        if (!lokal(req)) return tolak();
        const b = (await req.json().catch(() => ({}))) as { jenis?: string; nama?: string | null };
        const nama = b.nama ? String(b.nama) : null;
        const mode = b.jenis === "obrolan" ? "chat" : b.jenis === "pembaca" ? "pembaca" : null;
        if (!mode) return galat("jenis harus obrolan atau pembaca");
        if (nama && !daftarModel().some((m) => m.nama === nama && m.mode === mode)) return galat(`model ${mode === "chat" ? "obrolan" : "pembaca"} "${nama}" tidak ditemukan`);
        try {
          if (mode === "chat") mesin.pakaiGPT(nama); else mesin.pakaiPembaca(nama);
        } catch (e) { return galat((e as Error).message); }
        return Response.json(mesin.info());
      },
    },
    "/api/dataset": () => Response.json(daftarDataset()),
    "/api/perintah": () => Response.json(PERINTAH, { headers: {} }),
    "/api/job": {
      GET: () => Response.json(daftarJob().reverse()),
      POST: async (req) => {
        if (!lokal(req)) return tolak();
        const b = (await req.json().catch(() => ({}))) as { perintah?: string; opsi?: Record<string, unknown> };
        const j = mulaiJob(String(b.perintah ?? ""), b.opsi ?? {});
        return "galat" in j ? galat(j.galat) : Response.json(j);
      },
    },
    "/api/job/:id/log": (req) => {
      const r = bacaLog(req.params.id, Number(new URL(req.url).searchParams.get("dari") ?? 0));
      return r ? Response.json(r) : galat("log tidak ditemukan", 404);
    },
    "/api/job/:id/hentikan": { POST: (req) => (lokal(req) ? Response.json(hentikanJob(req.params.id)) : tolak()) },
    "/api/upload": {
      POST: async (req) => {
        if (!lokal(req)) return tolak();
        const form = await req.formData();
        const tujuan = String(form.get("tujuan") ?? "");
        if (!(tujuan in EKSTENSI)) return galat("tujuan harus aturan atau teks");
        const folder = tujuan === "aturan" ? LOKASI.dokumenUpload : LOKASI.teksSendiri;
        mkdirSync(folder, { recursive: true });
        const hasil: { nama: string; ukuran: number; lokasi: string }[] = [];
        for (const f of form.getAll("berkas")) {
          if (!(f instanceof File)) continue;
          const ext = extname(f.name).toLowerCase();
          if (!EKSTENSI[tujuan].includes(ext)) return galat(`${f.name}: jenis file tidak diizinkan untuk ${tujuan} (${EKSTENSI[tujuan].join(", ")})`);
          if (f.size > 50 * 1024 * 1024) return galat(`${f.name}: lebih dari 50 MB`);
          // nama file dibersihkan; jika sudah ada, ditambah angka (tidak menimpa)
          const dasar = basename(f.name, ext).replace(/[^\w.-]+/g, "_").slice(0, 80) || "berkas";
          let nama = dasar + ext, n = 1;
          while (existsSync(join(folder, nama))) nama = `${dasar}-${++n}${ext}`;
          await Bun.write(join(folder, nama), f);
          hasil.push({ nama, ukuran: f.size, lokasi: join(folder, nama) });
        }
        return Response.json({ ok: true, berkas: hasil, saran: tujuan === "aturan" ? "Latih ulang chatbot aturan (langkah 3) agar dokumen baru terbaca." : "Siapkan ulang dataset dengan sumber teks (langkah 2)." });
      },
    },
    "/api/info": () => Response.json(mesin.info()),
    "/api/chat": {
      POST: async (req) => {
        try {
          const b = (await req.json()) as { pesan?: string; riwayat?: GiliranRiwayat[]; topik?: Topik | null; mode?: Mode; suhu?: number };
          const pesan = String(b.pesan ?? "").trim().slice(0, 500);
          if (!pesan) return galat("Pesan kosong.");
          const mode: Mode = ["otomatis", "aturan", "wiki", "gpt", "penalaran", "agent", "rag"].includes(String(b.mode)) ? (b.mode as Mode) : "otomatis";
          const suhu = Math.min(1.5, Math.max(0.05, Number(b.suhu ?? 0.4)));
          const riwayat = Array.isArray(b.riwayat) ? b.riwayat.slice(-10).map((g) => ({ user: String(g.user ?? ""), bot: String(g.bot ?? ""), jalur: g.jalur })) : [];
          const t0 = performance.now();
          const r = mode === "agent"
            ? await jalankanModeAgent(mesin, pesan, { riwayat, topik: b.topik ?? null, suhu, internet: opsi["tanpa-internet"] !== "true" })
            : mesin.proses(pesan, { riwayat, topik: b.topik ?? null, mode, suhu });
          return Response.json({ ...r, totalMs: performance.now() - t0 });
        } catch (e) {
          return galat((e as Error).message, 500);
        }
      },
    },
    // Progres training/unduhan yang dijalankan manual di terminal (dibaca dari progres/*.json)
    "/api/progres": { GET: () => Response.json(bacaSemuaProgres()) },
    "/api/progres/hentikan": {
      POST: async (req) => {
        const b = (await req.json().catch(() => ({}))) as { jenis?: string; nama?: string };
        if (!lokal(req)) return tolak();
        if (!/^[\w-]{1,30}$/.test(String(b.jenis)) || !/^[\w.-]{1,60}$/.test(String(b.nama ?? ""))) return galat("jenis/nama tidak valid");
        return Response.json({ ok: mintaBerhenti(b.jenis as JenisProses, String(b.nama)) });
      },
    },
    "/api/memori": {
      GET: () => Response.json(mesin.memori.semua()),
      POST: async (req) => {
        const b = (await req.json().catch(() => ({}))) as { teks?: string };
        const teks = String(b.teks ?? "").trim();
        if (!teks) return galat("Teks memori kosong.");
        return Response.json(mesin.memori.tambah(teks));
      },
      DELETE: () => { mesin.memori.hapusSemua(); return Response.json({ ok: true }); },
    },
    "/api/export": {
      POST: async (req) => {
        const b = (await req.json().catch(() => ({}))) as { entri?: EntriEkspor[]; mode?: string; suhu?: number };
        const teks = eksporMarkdown(Array.isArray(b.entri) ? b.entri : [], mesin.info(), { mode: b.mode, suhu: b.suhu });
        return new Response(teks, { headers: { "content-type": "text/markdown; charset=utf-8" } });
      },
    },
    "/api/memori/:id": {
      DELETE: (req) => Response.json({ ok: mesin.memori.hapus(Number(req.params.id)) }),
    },
  },
  fetch: () => new Response("Tidak ditemukan", { status: 404 }),
});

const i = mesin.info();
console.log(`Chatbot web berjalan di http://localhost:${server.port}  ·  data & training: http://localhost:${server.port}/latih`);
console.log(`  aturan: ${i.aturan.pasal} pasal | Wikipedia: ${i.wiki ? `${i.wiki.artikel} artikel` : "belum ada"} | GPT: ${i.gpt?.nama ?? "belum ada"} | memori: ${i.memori} catatan`);
console.log("  Tekan Ctrl+C untuk berhenti.");

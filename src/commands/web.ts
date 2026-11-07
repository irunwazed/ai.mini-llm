/**
 * TAMPILAN WEB untuk chatbot + panel "cara kerja" (bahan belajar & uji coba).
 *
 * Pakai:  bun run web                 lalu buka http://localhost:3000
 *         bun run web -- --port 8080
 *
 * API:
 *   GET    /api/info          ringkasan model yang dimuat (aturan, Wikipedia, GPT, memori)
 *   POST   /api/chat          { pesan, riwayat?, topik?, mode?, suhu? } -> Respons (jawaban + proses + jejak GPT)
 *   GET    /api/memori        daftar catatan memori
 *   POST   /api/memori        { teks }  -> simpan catatan baru (manual oleh pengguna)
 *   DELETE /api/memori/:id    hapus satu catatan      DELETE /api/memori   hapus semua
 *   POST   /api/export        { entri[], mode, suhu } -> teks ekspor percakapan (markdown)
 *
 * Server tidak menyimpan riwayat percakapan: halaman mengirim riwayat & topik terakhir
 * di setiap pesan. Memori disimpan di server (data/memory.json).
 */
import { join } from "node:path";
import { eksporMarkdown, MesinChat, type EntriEkspor, type GiliranRiwayat, type Mode, type Topik } from "../chat-engine.js";
import { jalankanModeAgent } from "../agent/multi-mode.js";
import { argumen } from "./args.js";

const { folderModel, k, opsi } = argumen();
const mesin = new MesinChat({ folderModel, folderWiki: opsi.wiki, chatGpt: opsi["chat-gpt"], k, fileMemori: opsi.memori });
const port = Number(opsi.port ?? 3000);
const HALAMAN = join(import.meta.dir, "..", "web", "index.html");
const galat = (pesan: string, status = 400) => Response.json({ error: pesan }, { status });

const server = Bun.serve({
  port,
  routes: {
    "/": () => new Response(Bun.file(HALAMAN), { headers: { "content-type": "text/html; charset=utf-8" } }),
    "/api/info": () => Response.json(mesin.info()),
    "/api/chat": {
      POST: async (req) => {
        try {
          const b = (await req.json()) as { pesan?: string; riwayat?: GiliranRiwayat[]; topik?: Topik | null; mode?: Mode; suhu?: number };
          const pesan = String(b.pesan ?? "").trim().slice(0, 500);
          if (!pesan) return galat("Pesan kosong.");
          const mode: Mode = ["otomatis", "aturan", "wiki", "gpt", "penalaran", "agent"].includes(String(b.mode)) ? (b.mode as Mode) : "otomatis";
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
console.log(`Chatbot web berjalan di http://localhost:${server.port}`);
console.log(`  aturan: ${i.aturan.pasal} pasal | Wikipedia: ${i.wiki ? `${i.wiki.artikel} artikel` : "belum ada"} | GPT: ${i.gpt?.nama ?? "belum ada"} | memori: ${i.memori} catatan`);
console.log("  Tekan Ctrl+C untuk berhenti.");

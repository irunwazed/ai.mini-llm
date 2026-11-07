/**
 * CHATBOT interaktif di terminal. Logikanya sama persis dengan tampilan web (src/chat-engine.ts).
 *
 * Pakai:  bun run chat
 *         bun run chat -- --proses          (tampilkan juga langkah-langkah cara kerjanya)
 *         bun run chat -- --agent           (mulai dalam mode agent: semua mode + internet)
 *         bun run chat -- --agent --tanpa-internet
 * Perintah di dalam chat:
 *   :ingat teks   simpan catatan ke memori (mis. :ingat nama saya Irun)
 *   :memori       lihat semua catatan memori
 *   :lupa N       hapus catatan nomor N        :lupa semua   hapus semua memori
 *   :buka N       tampilkan teks lengkap sumber nomor N
 *   :gpt teks     lanjutkan teks dengan model GPT pretrain (opsi --gpt <nama>)
 *   :proses       nyalakan/matikan tampilan proses
 *   :agent        nyalakan/matikan mode agent (coba semua mode, cari di internet bila perlu)
 *   :export       ekspor percakapan ke teks (untuk disalin) dan file exports/chat-*.md
 *   :baru         mulai percakapan baru (riwayat dihapus, memori tetap)
 *   :bantuan      daftar perintah          :keluar   berhenti
 */
import { createInterface } from "node:readline";
import { mkdirSync, writeFileSync } from "node:fs";
import { eksporMarkdown, keEntriEkspor, MesinChat, type EntriEkspor, type GiliranRiwayat, type Respons, type Topik } from "../chat-engine.js";
import { Tokenizer } from "../gpt/bpe.js";
import { gptTerbaru, muatGPT } from "../gpt/checkpoint.js";
import { GPT } from "../gpt/model.js";
import { hasilkanTeks } from "../gpt/text-generator.js";
import { jalankanModeAgent } from "../agent/multi-mode.js";
import { argumen } from "./args.js";

const { folderModel, k, opsi } = argumen();
const mesin = new MesinChat({ folderModel, folderWiki: opsi.wiki, chatGpt: opsi["chat-gpt"], k, fileMemori: opsi.memori });
let tampilProses = opsi.proses === "true";
let modeAgent = opsi.agent === "true";
const internet = opsi["tanpa-internet"] !== "true";

const BANTUAN = [
  "Tanyakan tentang aturan, pengetahuan umum, atau sekadar menyapa. Pertanyaan lanjutan (\"kapan dia lahir?\") ikut memakai riwayat.",
  "  :ingat teks   simpan catatan ke memori, mis. :ingat nama saya Irun",
  "  :memori       lihat catatan memori        :lupa N / :lupa semua   hapus memori",
  "  :buka N       tampilkan teks lengkap sumber nomor N",
  "  :gpt teks     lanjutkan teks dengan model GPT pretrain (karangan model)",
  "  :proses       tampilkan/sembunyikan cara kerja",
  "  :agent        mode agent: coba semua mode, pilih yang paling cocok, cari di internet bila perlu",
  "  :export       ekspor percakapan (teks untuk disalin + file di exports/)",
  "  :baru         mulai percakapan baru (memori tetap tersimpan)",
  "  :keluar       berhenti",
].join("\n");

const i = mesin.info();
console.log(`Chatbot — aturan: ${i.aturan.pasal} pasal | Wikipedia: ${i.wiki ? `${i.wiki.artikel} artikel` : "belum ada (bun run index-wiki)"} | GPT obrolan: ${i.gpt ? i.gpt.nama : "belum ada"} | memori: ${i.memori} catatan`);
console.log(BANTUAN + "\n");

/** GPT pretrain untuk :gpt (dimuat saat pertama dipakai). */
let gptTeks: { nama: string; model: GPT; tok: Tokenizer } | null | undefined;
function muatGPTTeks() {
  if (gptTeks !== undefined) return gptTeks;
  const nama = opsi.gpt ?? gptTerbaru("teks");
  if (!nama) return (gptTeks = null);
  const { meta, P } = muatGPT(nama);
  return (gptTeks = { nama, model: new GPT(meta.config, P), tok: new Tokenizer(meta.bpe) });
}

const indentasi = (s: string) => s.split("\n").join("\n     ");
let terakhir: Respons | null = null;
let topik: Topik | null = null;
const riwayat: GiliranRiwayat[] = [];
const log: EntriEkspor[] = []; // seluruh percakapan sesi ini, untuk :export

function cetakProses(r: Respons) {
  console.log("     ── cara kerja ──");
  for (const l of r.proses) {
    console.log(`     • ${l.tahap}: ${l.hasil}${l.ms >= 1 ? ` (${l.ms.toFixed(0)} ms)` : ""}`);
    const d = l.data as { langkah?: { pikir: string; aksi: string; hasil: string; cek: string }[]; percobaan?: { suhu: number; teks: string; niat: string | null; diterima: boolean }[] } | undefined;
    if ((l.tahap === "Agent penalaran" || l.tahap === "Agent multi-mode") && d?.langkah) {
      d.langkah.forEach((a, i) => console.log(`       ${i + 1}. Pikir: ${a.pikir}\n          Aksi : ${a.aksi}\n          Hasil: ${a.hasil}\n          Cek  : ${a.cek}`));
    }
    if (d?.percobaan) {
      d.percobaan.forEach((p, i) => console.log(`       ${i + 1}. suhu ${p.suhu.toFixed(1)} ${p.diterima ? "✓" : "✗"} [${p.niat ?? "-"}] ${p.teks.slice(0, 80)}`));
    }
  }
  if (r.gpt) console.log(`     • token dibuat: ${r.gpt.langkah.map((s) => `${JSON.stringify(s.token)}(${(100 * s.p).toFixed(0)}%)`).join(" ")}`);
  console.log();
}

function perintah(masukan: string): boolean {
  if (masukan === ":bantuan") { console.log(BANTUAN + "\n"); return true; }
  if (masukan === ":baru") { topik = null; riwayat.length = 0; console.log("Percakapan baru dimulai (memori tetap tersimpan).\n"); return true; }
  if (masukan === ":agent") { modeAgent = !modeAgent; console.log(`Mode agent ${modeAgent ? `dinyalakan${internet ? " (boleh mencari di internet)" : " (tanpa internet)"}` : "dimatikan"}.
`); return true; }
  if (masukan === ":proses") { tampilProses = !tampilProses; console.log(`Tampilan proses ${tampilProses ? "dinyalakan" : "dimatikan"}.\n`); return true; }
  if (masukan === ":export") {
    if (!log.length) { console.log("Belum ada percakapan untuk diekspor.\n"); return true; }
    const teks = eksporMarkdown(log, mesin.info());
    mkdirSync("exports", { recursive: true });
    const file = `exports/chat-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}.md`;
    writeFileSync(file, teks);
    console.log(`\n${teks}\nTersimpan juga di ${file} — salin teks di atas untuk ditempel.\n`);
    return true;
  }
  const ingat = masukan.match(/^:ingat\s+(.+)$/s);
  if (ingat) { const c = mesin.memori.tambah(ingat[1]); console.log(`Tersimpan di memori sebagai #${c.id}: "${c.teks}"\n`); return true; }
  if (masukan === ":memori") {
    const semua = mesin.memori.semua();
    console.log(semua.length ? semua.map((c) => `  #${c.id}  ${c.teks}   (${new Date(c.waktu).toLocaleString("id-ID")})`).join("\n") + "\n" : "Memori masih kosong. Simpan dengan :ingat <teks>\n");
    return true;
  }
  const lupa = masukan.match(/^:lupa\s+(semua|\d+)$/);
  if (lupa) {
    if (lupa[1] === "semua") { mesin.memori.hapusSemua(); console.log("Semua memori dihapus.\n"); }
    else console.log(mesin.memori.hapus(Number(lupa[1])) ? `Memori #${lupa[1]} dihapus.\n` : `Memori #${lupa[1]} tidak ada.\n`);
    return true;
  }
  const gptCmd = masukan.match(/^:gpt\s+(.+)$/s);
  if (gptCmd) {
    const g = muatGPTTeks();
    if (!g) console.log("Belum ada model GPT pretrain. Latih dulu: bun run train-gpt -- --data data/wiki-id.txt --name wiki --keep-case\n");
    else {
      process.stdout.write(`\nGPT (${g.nama})> ${gptCmd[1]}`);
      hasilkanTeks(g.model, g.tok, gptCmd[1], 80, { suhu: 0.8, topK: 40, topP: 0.9, penaltiUlang: 1.15 }, (s) => process.stdout.write(s));
      console.log("\n     [Teks karangan model GPT, BUKAN dasar hukum]\n");
    }
    return true;
  }
  const buka = masukan.match(/^:buka\s+(\d+)$/);
  if (buka) {
    const s = terakhir?.sumber[Number(buka[1]) - 1];
    console.log(s ? `=== ${s.label} ===\n\n${s.teks}\n` : "Nomor sumber tidak ada.\n");
    return true;
  }
  return false;
}

const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: "Kamu> " });
rl.prompt();
let sibuk = false;
rl.on("line", async (baris) => {
  if (sibuk) return;
  const masukan = baris.trim();
  if (!masukan) return rl.prompt();
  if (masukan === ":keluar" || masukan === ":q") return rl.close();
  if (masukan.startsWith(":") && perintah(masukan)) return rl.prompt();

  const t0 = performance.now();
  sibuk = true;
  const r = modeAgent ? await jalankanModeAgent(mesin, masukan, { riwayat, topik, internet }) : mesin.proses(masukan, { riwayat, topik });
  sibuk = false;
  log.push(keEntriEkspor(masukan, { ...r, totalMs: performance.now() - t0 }));
  terakhir = r;
  topik = r.topik;
  riwayat.push({ user: masukan, bot: r.teks, jalur: r.jalur });
  if (riwayat.length > 10) riwayat.shift();
  console.log("\nBot> " + indentasi(r.teks) + (r.label ? `\n     (${r.label})` : "") + "\n");
  if (tampilProses) cetakProses(r);
  rl.prompt();
});
rl.on("close", () => console.log("\nSampai jumpa."));

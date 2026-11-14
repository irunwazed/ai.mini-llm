/**
 * UJI GPT OBROLAN: apakah model MEMAKAI KONTEKS percakapan?
 *
 * Percakapan uji dibuat dari skenario yang sama dengan data latih (src/gpt/chat-context-data.ts), tetapi
 * dengan NILAI UJI (nama, kota, pekerjaan, hobi, makanan, nama hewan) yang tidak pernah dilihat saat latihan.
 * Balasan dianggap benar jika memuat nilai yang diharapkan (mis. kota yang disebut 3 giliran sebelumnya).
 *
 *   dengan konteks : riwayat + kartu konteks diberikan seperti biasa
 *   ABLASI         : riwayat dan kartu dihapus, hanya pertanyaan terakhir. Jika akurasi tetap tinggi,
 *                    model menebak/menghafal; jika turun drastis, model benar-benar membaca konteks.
 *   perplexity     : seberapa "terkejut" model oleh jawaban bot yang benar (makin kecil makin baik)
 *
 * Pakai:  bun run eval-konteks                         (model chat terbaru)
 *         bun run eval-konteks -- --name chat2 --banding chat --jumlah 300
 */
import { Tokenizer } from "../gpt/bpe.js";
import { balas, type Giliran } from "../gpt/chat.js";
import { buatPercakapanKonteks, type PercakapanKonteks } from "../gpt/chat-context-data.js";
import { gptTerbaru, muatGPT } from "../gpt/checkpoint.js";
import { dataChat } from "../gpt/dataset.js";
import { GPT } from "../gpt/model.js";
import { argumen } from "./args.js";

const { opsi } = argumen();
const jumlah = Number(opsi.jumlah ?? 300);
const uji = buatPercakapanKonteks(jumlah, true, Number(opsi.seed ?? 99));

interface Hasil { nama: string; per: Map<string, { n: number; benar: number; ablasi: number }>; ppl: number; salah: string[]; ms: number; kartu: boolean }

function ukur(nama: string): Hasil {
  const { meta, P } = muatGPT(nama);
  const model = new GPT(meta.config, P), tok = new Tokenizer(meta.bpe);
  const kartuDidukung = tok.idToken("<|konteks|>") !== undefined;
  const per = new Map<string, { n: number; benar: number; ablasi: number }>();
  const salah: string[] = [];
  let lossJumlah = 0, lossN = 0;
  const t0 = performance.now();
  for (const p of uji) {
    const kartu = p.turns[0].konteks ?? null;
    const isi = p.turns.filter((g) => g.konteks === undefined);
    const pertanyaan = isi.at(-2)!.user!, riwayat: Giliran[] = [];
    for (let i = 0; i + 1 < isi.length - 2; i += 2) riwayat.push({ user: isi[i].user!, bot: isi[i + 1].bot! });
    const deterministik = { suhu: 0.01, topK: 1, seed: 1, maksToken: 40 };
    const jawab = balas(model, tok, riwayat, pertanyaan, { ...deterministik, kartu }, meta.hurufKecil);
    const tanpa = balas(model, tok, [], pertanyaan, deterministik, meta.hurufKecil);
    const cocok = (t: string) => t.toLowerCase().includes(p.harapan.toLowerCase());
    const x = per.get(p.skenario) ?? { n: 0, benar: 0, ablasi: 0 };
    x.n++; if (cocok(jawab)) x.benar++; else if (salah.length < 6) salah.push(`  [${p.skenario}] "${pertanyaan}" -> "${jawab}" (harusnya memuat "${p.harapan}")`);
    if (cocok(tanpa)) x.ablasi++;
    per.set(p.skenario, x);
    // perplexity jawaban bot (semua giliran bot di percakapan ini)
    const { train } = dataChat(tok, [p.turns as never], model.c.context);
    for (const u of train) { lossJumlah += model.langkah(u.x, u.y, null, 1); lossN += u.y.filter((v) => v >= 0).length; }
  }
  return { nama, per, ppl: Math.exp(lossJumlah / Math.max(1, lossN)), salah, ms: (performance.now() - t0) / uji.length, kartu: kartuDidukung };
}

const nama = opsi.name ?? gptTerbaru("chat");
if (!nama) { console.error("Belum ada model GPT obrolan. Latih dulu: bun run siapkan-data -- --sumber obrolan --nama obrolan-v1 --tokenizer-dari wiki, lalu bun run train-gpt -- --dataset obrolan-v1 --from wiki --name chat"); process.exit(1); }
const daftar = [nama, ...(opsi.banding ? [opsi.banding] : [])].map(ukur);

console.log(`${uji.length} percakapan uji (nilai slot yang tidak pernah dilatih). Benar = balasan memuat nilai yang diharapkan.\n`);
const skenario = [...new Set(uji.map((p: PercakapanKonteks) => p.skenario))];
const kepala = daftar.map((h) => `${h.nama.padEnd(8)} konteks / ablasi`).join(" | ");
console.log(`${"skenario".padEnd(15)} |   n | ${kepala}`);
for (const s of skenario) {
  const n = daftar[0].per.get(s)?.n ?? 0;
  const kol = daftar.map((h) => { const x = h.per.get(s)!; return `${`${Math.round((100 * x.benar) / x.n)}%`.padStart(8)} / ${`${Math.round((100 * x.ablasi) / x.n)}%`.padStart(6)}`; });
  console.log(`${s.padEnd(15)} | ${String(n).padStart(3)} | ${kol.join(" | ")}`);
}
const total = (h: Hasil, k: "benar" | "ablasi") => [...h.per.values()].reduce((a, x) => a + x[k], 0);
console.log(`${"TOTAL".padEnd(15)} | ${String(uji.length).padStart(3)} | ${daftar.map((h) => `${`${Math.round((100 * total(h, "benar")) / uji.length)}%`.padStart(8)} / ${`${Math.round((100 * total(h, "ablasi")) / uji.length)}%`.padStart(6)}`).join(" | ")}`);
for (const h of daftar) {
  console.log(`\n${h.nama}: perplexity jawaban bot ${h.ppl.toFixed(2)} · ${h.ms.toFixed(0)} ms/percakapan · kartu konteks ${h.kartu ? "didukung" : "TIDAK didukung (model lama)"}`);
  if (h.salah.length) console.log(`Contoh salah:\n${h.salah.join("\n")}`);
}

/**
 * PINDAHKAN DATA ke struktur folder baru (sekali saja):
 *   data/wiki-id.txt          -> data/mentah/wikipedia/wiki-id.txt
 *   data/wiki-id.sources.txt  -> data/mentah/wikipedia/sumber.txt
 *   data/chat-id.jsonl        -> data/mentah/obrolan/chat-id.jsonl
 *   data/reader-id.jsonl      -> data/mentah/pembaca/reader-id.jsonl
 * File yang masih ditulis proses lain (diubah < 2 menit lalu, mis. fetch-data berjalan) tidak dipindah,
 * kecuali --paksa. Model yang sudah dilatih tidak berubah.
 *
 * Pakai:  bun run migrasi-data
 *         bun run migrasi-data -- --kering     (tampilkan rencana saja)
 */
import { existsSync, mkdirSync, renameSync, statSync } from "node:fs";
import { dirname } from "node:path";
import { LOKASI, LOKASI_LAMA } from "../paths.js";
import { argumen } from "./args.js";

const { opsi } = argumen();
const kering = opsi.kering === "true", paksa = opsi.paksa === "true";
let dipindah = 0, ditunda = 0;
for (const [lama, baru] of LOKASI_LAMA) {
  if (!existsSync(lama)) { console.log(`  · ${lama}: tidak ada${existsSync(baru) ? ` (sudah di ${baru})` : ""}`); continue; }
  if (existsSync(baru)) { console.log(`  ! ${lama}: dilewati, ${baru} sudah ada. Gabungkan/hapus salah satu secara manual.`); ditunda++; continue; }
  const umur = (Date.now() - statSync(lama).mtimeMs) / 1000;
  if (umur < 120 && !paksa) {
    console.log(`  ! ${lama}: diubah ${Math.round(umur)} detik lalu, kemungkinan masih ditulis proses lain (mis. fetch-data). Hentikan dulu, atau pakai --paksa.`);
    ditunda++; continue;
  }
  console.log(`  ${kering ? "→" : "✓"} ${lama}  ->  ${baru}`);
  if (!kering) { mkdirSync(dirname(baru), { recursive: true }); renameSync(lama, baru); dipindah++; }
}
for (const d of [LOKASI.teksSendiri, LOKASI.siap, LOKASI.dokumenUpload, LOKASI.log]) if (!kering) mkdirSync(d, { recursive: true });
console.log(kering ? "\n(kering: tidak ada yang dipindah)" : `\n${dipindah} file dipindah${ditunda ? `, ${ditunda} ditunda` : ""}. Folder baru: ${LOKASI.mentah}/, ${LOKASI.siap}/`);
if (!kering && dipindah) console.log("Langkah berikutnya: bun run siapkan-data (lihat PANDUAN.md).");

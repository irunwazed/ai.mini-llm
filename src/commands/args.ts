/**
 * Parser argumen sederhana:  --kunci nilai,  --flag (tanpa nilai = "true"),  dan sisa teks bebas.
 */
const FLAG = new Set(["pre-ln", "resume", "tanpa-embedding", "keep-case", "stop-newline", "proses", "agent", "tanpa-internet"]);

export function argumen(argv = process.argv.slice(2)) {
  const opsi: Record<string, string> = {};
  const bebas: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      const kunci = argv[i].slice(2);
      const nilai = argv[i + 1];
      if (FLAG.has(kunci) || nilai === undefined || nilai.startsWith("--")) opsi[kunci] = "true";
      else { opsi[kunci] = nilai; i++; }
    } else bebas.push(argv[i]);
  }
  return {
    opsi,
    teks: bebas.join(" "),
    folderModel: opsi.model ?? "model",
    folderDokumen: opsi.dokumen ?? "documents",
    k: Number(opsi.k ?? 3), // jumlah sumber yang ditampilkan
  };
}

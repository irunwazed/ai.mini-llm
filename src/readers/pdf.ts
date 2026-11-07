/**
 * PDF -> teks per halaman. Baris dibentuk ulang dari posisi Y setiap potongan teks,
 * supaya heading "Pasal 5" tetap berada di baris sendiri.
 */
import { getDocumentProxy } from "unpdf";

export async function bacaPdf(buf: Buffer): Promise<string[]> {
  const pdf = await getDocumentProxy(new Uint8Array(buf));
  const halaman: string[] = [];
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const isi = await page.getTextContent();
    let teks = "";
    let yLama: number | null = null;
    for (const item of isi.items) {
      if (!("str" in item)) continue;
      const y = item.transform[5];
      if (yLama !== null && Math.abs(y - yLama) > 2 && !teks.endsWith("\n")) teks += "\n";
      teks += item.str;
      if (item.hasEOL) teks += "\n";
      yLama = y;
    }
    halaman.push(teks);
    page.cleanup();
  }
  await pdf.cleanup();
  return halaman;
}

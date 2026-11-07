/** DOCX -> teks polos (satu paragraf per baris). */
import mammoth from "mammoth";

export async function bacaDocx(buf: Buffer): Promise<string[]> {
  const { value } = await mammoth.extractRawText({ buffer: buf });
  return [value.replace(/\n{2,}/g, "\n")];
}

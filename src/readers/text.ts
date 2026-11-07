/** TXT / Markdown -> teks. BOM UTF-8 dibuang. */
export function bacaTeks(buf: Buffer): string[] {
  return [buf.toString("utf8").replace(/^﻿/, "")];
}

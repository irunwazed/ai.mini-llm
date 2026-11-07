/**
 * PERCAKAPAN dengan GPT yang sudah di-fine-tune untuk chat.
 *
 * Prompt disusun dari riwayat dengan token khusus, persis seperti saat training:
 *   <|user|> halo<|end|><|bot|> Halo! Ada yang bisa saya bantu?<|end|><|user|> apa kabar?<|end|><|bot|>
 * Model lalu melanjutkan teks setelah <|bot|> sampai menghasilkan <|end|>.
 * Jika riwayat terlalu panjang untuk konteks, giliran paling awal dibuang.
 */
import { BOT, END, UNK, USER, type Tokenizer } from "./bpe.js";
import type { GPT } from "./model.js";
import { simulasiGPT, type SimulasiGPT } from "./inspect.js";
import { hasilkanDariId } from "./text-generator.js";

export interface Giliran { user: string; bot: string }

export interface OpsiChat { suhu?: number; topP?: number; topK?: number; penaltiUlang?: number; maksToken?: number; seed?: number; simulasi?: boolean }

export function balas(model: GPT, tok: Tokenizer, riwayat: Giliran[], pesan: string, opsi: OpsiChat = {}, hurufKecil = false): string {
  return balasDetail(model, tok, riwayat, pesan, opsi, hurufKecil).teks;
}

/** Jejak pembuatan balasan, untuk belajar: token prompt dan pilihan token di setiap langkah. */
export interface JejakGPT {
  prompt: string[];                                    // potongan token BPE yang dibaca model
  langkah: { token: string; p: number; kandidat: { token: string; p: number }[] }[];
  suhu: number; topK: number; topP: number; penaltiUlang: number;
  simulasi?: SimulasiGPT;                              // angka-angka di dalam model (opsi.simulasi)
}

export function balasDetail(
  model: GPT, tok: Tokenizer, riwayat: Giliran[], pesan: string, opsi: OpsiChat = {}, hurufKecil = false,
): { teks: string; jejak: JejakGPT } {
  const idUser = tok.idToken(USER)!, idBot = tok.idToken(BOT)!, idEnd = tok.idToken(END)!;
  const norm = (s: string) => " " + (hurufKecil ? s.toLowerCase() : s).trim();
  const maks = opsi.maksToken ?? 60;
  const giliran = (g: Giliran) => [idUser, ...tok.encode(norm(g.user)), idEnd, idBot, ...tok.encode(norm(g.bot)), idEnd];
  const akhir = [idUser, ...tok.encode(norm(pesan)), idEnd, idBot];

  // Pangkas riwayat dari depan agar prompt + jawaban muat dalam konteks
  let r = riwayat.slice();
  const panjang = () => r.reduce((s, g) => s + giliran(g).length, 0) + akhir.length;
  while (r.length && panjang() + maks > model.c.context) r = r.slice(1);
  const prompt = [...r.flatMap(giliran), ...akhir].slice(-(model.c.context - 1));

  const larang = [idUser, idBot, tok.idToken(UNK)].filter((x): x is number => x !== undefined);
  const jejak: JejakGPT = {
    prompt: prompt.map((id) => tok.vocab[id]),
    langkah: [],
    suhu: opsi.suhu ?? 0.7, topP: opsi.topP ?? 0.9, topK: opsi.topK ?? 40, penaltiUlang: opsi.penaltiUlang ?? 1.15,
  };
  const { teks, ids } = hasilkanDariId(model, tok, prompt.slice(), maks, {
    suhu: jejak.suhu, topP: jejak.topP, topK: jejak.topK, penaltiUlang: jejak.penaltiUlang,
    larang, seed: opsi.seed, berhentiDi: (_, id) => id === idEnd,
    catat: (kandidat, dipilih) => jejak.langkah.push({
      token: tok.vocab[dipilih],
      p: kandidat.find((k) => k.id === dipilih)?.p ?? 0,
      kandidat: kandidat.map((k) => ({ token: tok.vocab[k.id], p: k.p })),
    }),
  });
  if (opsi.simulasi) jejak.simulasi = simulasiGPT(model, tok, prompt, ids, jejak.suhu);
  return { teks: rapikanBalasan(teks), jejak };
}

/** Huruf kapital di awal kalimat (berguna jika model dilatih dengan huruf kecil). */
export function rapikanBalasan(s: string): string {
  return s.trim().replace(/(^|[.!?]\s+)([a-z])/g, (_, a, b) => a + b.toUpperCase());
}

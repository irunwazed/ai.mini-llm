/**
 * LOGIN GOOGLE (OAuth 2.0 + PKCE) untuk Google Drive — tanpa library, hanya fetch.
 *
 * Scope: drive.file -> aplikasi HANYA bisa melihat/mengubah file yang ia buat sendiri (folder "LLM dari Nol").
 * File lain di Drive Anda tidak bisa dibaca aplikasi ini.
 *
 * Persiapan (sekali): Google Cloud Console -> buat project -> aktifkan "Google Drive API" -> OAuth consent
 * screen (External, tambahkan email Anda sebagai test user) -> Credentials -> OAuth client ID tipe
 * "Desktop app" -> Download JSON -> simpan sebagai .rahasia/google-oauth.json
 * (atau isi env GOOGLE_CLIENT_ID dan GOOGLE_CLIENT_SECRET).
 *
 * Token disimpan di .rahasia/google-token.json (di-gitignore). Token TIDAK pernah dicetak.
 */
import { createHash, randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { LOKASI } from "../paths.js";

export const SCOPE = "https://www.googleapis.com/auth/drive.file";
export const FILE_KLIEN = join(LOKASI.rahasia, "google-oauth.json");
export const FILE_TOKEN = join(LOKASI.rahasia, "google-token.json");
const URL_AUTH = "https://accounts.google.com/o/oauth2/v2/auth";
const URL_TOKEN = "https://oauth2.googleapis.com/token";
const URL_CABUT = "https://oauth2.googleapis.com/revoke";

export interface KlienOAuth { client_id: string; client_secret: string }
interface Token { access_token: string; refresh_token: string; expires_at: number; scope: string; dibuat: string }

/** Client OAuth dari env atau .rahasia/google-oauth.json (format unduhan Console: { installed: {...} }). */
export function bacaKlien(): KlienOAuth | null {
  if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
    return { client_id: process.env.GOOGLE_CLIENT_ID, client_secret: process.env.GOOGLE_CLIENT_SECRET };
  }
  if (!existsSync(FILE_KLIEN)) return null;
  try {
    const j = JSON.parse(readFileSync(FILE_KLIEN, "utf8"));
    const k = j.installed ?? j.web ?? j;
    return k.client_id && k.client_secret ? { client_id: k.client_id, client_secret: k.client_secret } : null;
  } catch { return null; }
}

/**
 * Simpan JSON client OAuth hasil unduhan Google Cloud Console (dari halaman popup login).
 * Hanya bentuk { installed|web: { client_id, client_secret } } yang diterima.
 */
export function simpanKlien(teks: string, ganti = false): { ok: true } | { galat: string } {
  if (teks.length > 10_000) return { galat: "File terlalu besar untuk JSON client OAuth." };
  let j: { installed?: Record<string, unknown>; web?: Record<string, unknown> };
  try { j = JSON.parse(teks); } catch { return { galat: "Bukan file JSON." }; }
  const k = j.installed ?? j.web;
  const id = typeof k?.client_id === "string" ? k.client_id : "", rahasia = typeof k?.client_secret === "string" ? k.client_secret : "";
  if (!id.endsWith(".apps.googleusercontent.com") || !rahasia) return { galat: "Bukan JSON client OAuth Google (harus berisi installed.client_id …apps.googleusercontent.com dan client_secret). Pilih tipe Desktop app lalu Download JSON." };
  if (existsSync(FILE_KLIEN) && !ganti) return { galat: "Client OAuth sudah ada. Centang \"ganti\" untuk menimpa." };
  mkdirSync(LOKASI.rahasia, { recursive: true });
  writeFileSync(FILE_KLIEN, JSON.stringify({ installed: { client_id: id, client_secret: rahasia } }, null, 1), { mode: 0o600 });
  return { ok: true };
}

function bacaToken(): Token | null {
  if (!existsSync(FILE_TOKEN)) return null;
  try { return JSON.parse(readFileSync(FILE_TOKEN, "utf8")) as Token; } catch { return null; }
}
function simpanToken(t: Token): void {
  mkdirSync(LOKASI.rahasia, { recursive: true });
  writeFileSync(FILE_TOKEN, JSON.stringify(t, null, 1), { mode: 0o600 });
}

export const PETUNJUK_KLIEN = `Belum ada OAuth client Google. Sekali saja:
  1. https://console.cloud.google.com/ -> buat project -> "APIs & Services" -> aktifkan "Google Drive API"
  2. "OAuth consent screen": External, isi nama aplikasi, tambahkan email Anda sebagai Test user
  3. "Credentials" -> Create credentials -> OAuth client ID -> Application type: Desktop app
  4. Download JSON -> simpan sebagai ${FILE_KLIEN}
     (atau set env GOOGLE_CLIENT_ID dan GOOGLE_CLIENT_SECRET)`;

/** Ringkasan untuk status (tanpa isi token). */
export function statusLogin(): { klien: boolean; login: boolean; scope: string | null; sejak: string | null } {
  const t = bacaToken();
  return { klien: !!bacaKlien(), login: !!t?.refresh_token, scope: t?.scope ?? null, sejak: t?.dibuat ?? null };
}

const b64url = (b: Buffer) => b.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const tertunda = new Map<string, { verifier: string; redirect: string; waktu: number }>();

/** Buat URL login Google. `redirect` = alamat loopback yang akan menerima ?code=&state= */
export function buatUrlLogin(redirect: string): { url: string; state: string } {
  const klien = bacaKlien();
  if (!klien) throw new Error(PETUNJUK_KLIEN);
  const verifier = b64url(randomBytes(48));
  const challenge = b64url(createHash("sha256").update(verifier).digest());
  const state = b64url(randomBytes(16));
  for (const [k, v] of tertunda) if (Date.now() - v.waktu > 15 * 60_000) tertunda.delete(k);
  tertunda.set(state, { verifier, redirect, waktu: Date.now() });
  const q = new URLSearchParams({
    client_id: klien.client_id, redirect_uri: redirect, response_type: "code", scope: SCOPE,
    code_challenge: challenge, code_challenge_method: "S256", state, access_type: "offline", prompt: "consent",
  });
  return { url: `${URL_AUTH}?${q}`, state };
}

export function adaLoginTertunda(state: string | null): boolean { return !!state && tertunda.has(state); }

/** Tukar ?code= dari Google menjadi token, lalu simpan. */
export async function selesaikanLogin(state: string, code: string): Promise<void> {
  const t = tertunda.get(state);
  if (!t) throw new Error("Sesi login tidak dikenal atau kedaluwarsa. Ulangi login.");
  tertunda.delete(state);
  const klien = bacaKlien()!;
  const r = await fetch(URL_TOKEN, {
    method: "POST", headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ code, client_id: klien.client_id, client_secret: klien.client_secret, redirect_uri: t.redirect, grant_type: "authorization_code", code_verifier: t.verifier }),
  });
  const j = (await r.json()) as { access_token?: string; refresh_token?: string; expires_in?: number; scope?: string; error?: string; error_description?: string };
  if (!r.ok || !j.access_token) throw new Error(`Login gagal: ${j.error ?? r.status} ${j.error_description ?? ""}`.trim());
  if (!j.refresh_token) throw new Error("Google tidak memberi refresh token. Cabut akses aplikasi di https://myaccount.google.com/permissions lalu login ulang.");
  simpanToken({ access_token: j.access_token, refresh_token: j.refresh_token, expires_at: Date.now() + (j.expires_in ?? 3600) * 1000, scope: j.scope ?? SCOPE, dibuat: new Date().toISOString() });
}

let menyegarkan: Promise<string> | null = null;
/** Access token yang masih berlaku (diperbarui otomatis dari refresh token). */
export async function aksesToken(): Promise<string> {
  const t = bacaToken();
  if (!t?.refresh_token) throw new Error("Belum login Google Drive. Jalankan: bun run drive -- --login  (atau tombol Login di halaman /latih)");
  if (t.expires_at - 60_000 > Date.now()) return t.access_token;
  menyegarkan ??= (async () => {
    try {
      const klien = bacaKlien();
      if (!klien) throw new Error(PETUNJUK_KLIEN);
      const r = await fetch(URL_TOKEN, {
        method: "POST", headers: { "content-type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ client_id: klien.client_id, client_secret: klien.client_secret, refresh_token: t.refresh_token, grant_type: "refresh_token" }),
      });
      const j = (await r.json()) as { access_token?: string; expires_in?: number; error?: string };
      if (!r.ok || !j.access_token) {
        throw new Error(j.error === "invalid_grant" ? "Login Google kedaluwarsa/dicabut. Login ulang: bun run drive -- --login" : `Gagal memperbarui token: ${j.error ?? r.status}`);
      }
      simpanToken({ ...t, access_token: j.access_token, expires_at: Date.now() + (j.expires_in ?? 3600) * 1000 });
      return j.access_token;
    } finally { menyegarkan = null; }
  })();
  return menyegarkan;
}

/** Cabut token di Google lalu hapus file token lokal. */
export async function logout(): Promise<boolean> {
  const t = bacaToken();
  if (!t) return false;
  await fetch(URL_CABUT, { method: "POST", headers: { "content-type": "application/x-www-form-urlencoded" }, body: new URLSearchParams({ token: t.refresh_token }) }).catch(() => null);
  rmSync(FILE_TOKEN, { force: true });
  return true;
}

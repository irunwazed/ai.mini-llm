/**
 * Halaman kecil untuk JENDELA POPUP login Google Drive (dibuka dari /latih).
 *   - persiapan: belum ada OAuth client -> langkah-langkah + upload file JSON client
 *   - selesai: hasil login -> kabari jendela /latih (postMessage, tanpa token) lalu tutup sendiri
 */
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

const GAYA = `<style>
  body { font: 14px/1.55 system-ui, "Segoe UI", sans-serif; margin: 0; padding: 22px 24px; color: #1d1d1b; background: #f6f5f1; }
  h2 { font-size: 18px; margin: 0 0 10px; } ol { padding-left: 20px; } li { margin: 5px 0; }
  .kotak { background: #fff; border: 1px solid #e3e1d9; border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
  button { font: inherit; padding: 6px 14px; border-radius: 6px; border: 1px solid #1d1d1b; background: #1d1d1b; color: #fff; cursor: pointer; }
  .galat { color: #b5442e; } .ok { color: #1f6f3f; } .kecil { font-size: 12.5px; color: #6b6a64; } code { background: #eeece4; padding: 1px 4px; border-radius: 4px; }
  a { color: #2f5fa7; }
</style>`;

export function halamanPersiapan(): string {
  return `<!doctype html><html lang="id"><head><meta charset="utf-8"><title>Hubungkan Google Drive</title>${GAYA}</head><body>
<h2>Hubungkan Google Drive (sekali saja)</h2>
<div class="kecil">Aplikasi ini butuh "OAuth client" milik Anda sendiri, agar login langsung dari komputer Anda ke Google, tanpa perantara.</div>
<div class="kotak"><ol>
  <li>Buka <a href="https://console.cloud.google.com/projectcreate" target="_blank" rel="noopener">Google Cloud Console</a> dan buat project (mis. <code>llm-dari-nol</code>).</li>
  <li>Aktifkan <a href="https://console.cloud.google.com/apis/library/drive.googleapis.com" target="_blank" rel="noopener">Google Drive API</a>.</li>
  <li><a href="https://console.cloud.google.com/apis/credentials/consent" target="_blank" rel="noopener">OAuth consent screen</a>: pilih <b>External</b>, isi nama aplikasi, lalu tambahkan email Anda di <b>Test users</b>.</li>
  <li><a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noopener">Credentials</a> → <b>Create credentials → OAuth client ID</b> → Application type <b>Desktop app</b> → <b>Download JSON</b>.</li>
  <li>Pilih file JSON itu di bawah.</li>
</ol></div>
<div class="kotak">
  <input type="file" id="berkas" accept=".json,application/json">
  <label class="kecil"><input type="checkbox" id="ganti"> ganti client yang sudah ada</label>
  <div id="pesan" class="kecil" style="margin-top:8px">File disimpan hanya di komputer ini (<code>.rahasia/google-oauth.json</code>).</div>
</div>
<script>
document.getElementById("berkas").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const p = document.getElementById("pesan");
  p.className = "kecil"; p.textContent = "Menyimpan…";
  const r = await fetch("/api/drive/klien" + (document.getElementById("ganti").checked ? "?ganti=1" : ""), { method: "POST", headers: { "content-type": "application/json" }, body: await f.text() });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { p.className = "galat"; p.textContent = j.error || "Gagal menyimpan."; return; }
  p.className = "ok"; p.textContent = "Tersimpan. Membuka halaman login Google…";
  setTimeout(() => location.replace("/drive/login"), 600);
};
</script></body></html>`;
}

export function halamanSelesai(hasil: string): string {
  const ok = hasil === "berhasil";
  return `<!doctype html><html lang="id"><head><meta charset="utf-8"><title>Google Drive</title>${GAYA}</head><body>
<h2 class="${ok ? "ok" : "galat"}">${ok ? "✓ Terhubung ke Google Drive" : "Login Google Drive gagal"}</h2>
<div class="kotak">${ok ? "Jendela ini menutup sendiri…" : esc(hasil)}</div>
<button onclick="tutup()">${ok ? "Tutup" : "Tutup & coba lagi dari halaman Latih"}</button>
<script>
const hasil = ${JSON.stringify(hasil).replace(/</g, "\\u003c").replace(/>/g, "\\u003e")};
function tutup() {
  if (window.opener && !window.opener.closed) window.opener.postMessage({ jenis: "drive-login", hasil }, "*");
  // halaman Latih juga memeriksa status login sendiri, jadi cukup menutup jendela ini
  window.close();
  // jika browser tidak mengizinkan menutup (mis. bukan dibuka sebagai popup): tawarkan kembali ke halaman Latih
  setTimeout(() => { document.querySelector(".kotak").innerHTML = 'Silakan tutup jendela ini, atau <a href="/latih#drive">kembali ke halaman Latih</a>.'; }, 400);
}
if (hasil === "berhasil") setTimeout(tutup, 900);
else if (window.opener && !window.opener.closed) window.opener.postMessage({ jenis: "drive-login", hasil }, "*");
</script></body></html>`;
}

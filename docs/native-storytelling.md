# Native storytelling, animated subtitles, dan multi-source ingest

Workflow ini ditujukan untuk video yang editing-nya sudah selesai, termasuk video portrait. Tetap memakai CLI, analisis visual, narration generator, TTS, dan FFmpeg existing.

```text
Local video / public video URL
  → ingest + source metadata
  → scene/keyframe analysis (tanpa mengubah footage full)
  → narration plan + storytelling
  → TTS + measured duration + bounded rewrite bila perlu
  → word timing → SRT + ASS
  → audio mixing + final MP4
```

## Menjalankan video native

```bash
dubbingstory run --input video.mp4 --project cerita_native --mode full --lang id --engine edge --ratio source --subtitle-style karaoke --burn-subtitles
```

- `--mode full --ratio source` mempertahankan urutan, durasi, dan komposisi video. Scene detection tetap dipakai untuk analisis.
- `--subtitle-style word` menampilkan satu kata pada waktunya; `karaoke` menampilkan frasa dengan kata aktif disorot.
- `--no-burn-subtitles` mengekspor subtitle tanpa overlay ke video. SRT tetap tersedia pada kedua style.
- Native/animated output otomatis memakai `--narration-fit rewrite` jika tidak dipilih. Rewrite hanya pada beat yang terlalu panjang, maksimal `tts.fit_retries`, kemudian TTS diukur kembali. Gunakan `--narration-fit error` untuk berhenti tanpa rewrite.
- Strict fitting tidak otomatis mempercepat audio atau video. Parameter `tts.max_speedup` tetap berlaku pada jalur `legacy`.
- Jika narasi masih tidak muat, proses gagal dengan identitas beat. Audio canvas tidak dipakai untuk memotong ucapan yang berlebih.

Tanpa filter visual, video dengan codec yang kompatibel MP4 dapat memakai stream copy. Burn subtitle membutuhkan encoding; semua filter visual digabungkan dalam satu tahap encoding video, tanpa scaling pada `source`. Encoding HDR sengaja ditolak sampai ada kebijakan color management yang eksplisit; stream copy HDR tetap dapat digunakan bila codec kompatibel dan burn dimatikan.

Pilihan `source` pada `--mode summary` mempertahankan **geometri**, sedangkan summary tetap memilih/memotong footage seperti sebelumnya. Gunakan `full` untuk mempertahankan seluruh video asli.

Timeline memakai durasi stream video, bukan durasi container yang dapat lebih panjang karena audio. Jika durasi stream tidak tersedia, timestamp paket video diperiksa. Input dengan durasi akhir yang tidak dapat ditentukan secara pasti ditolak pada mode strict, termasuk VFR tanpa durasi paket akhir.

## Ingest URL dan hak penggunaan

```bash
dubbingstory ingest --url "https://www.rednote.com/discovery/item/6a47c2cc000000001700a073" --i-have-rights --quality max --project rednote_sample
```

Input yang didukung adapter adalah URL video YouTube, Douyin, Xiaohongshu/RedNote, dan share link yang dapat diselesaikan melalui redirect HTTP biasa. Share text harus memuat tepat satu URL. Profile, playlist, live stream, dan post gambar/carousel tidak didukung.

`--quality max` menghindari batas tinggi piksel dan memilih rendition terbaik sesuai filter codec existing. Arti `720`, `1080`, `2k`, dan `4k` tetap batas **tinggi**; `1080` dapat mengecualikan video portrait 1080×1920. AV1 tetap dikecualikan oleh selector existing.

`--i-have-rights` tetap wajib untuk URL. Downloader tidak melewati DRM, login, CAPTCHA, atau proteksi platform. Konfigurasi cookie existing hanya diterapkan pada YouTube, bukan otomatis ke provider baru. Kegagalan dibedakan menjadi URL tidak didukung, akses dibatasi, video tidak tersedia, jaringan, media tidak valid, atau kegagalan extractor. Tidak semua share link menghasilkan redirect yang dapat diproses tanpa browser.

Nama project URL otomatis sekarang mencakup provider/video identity. Gunakan `--project` untuk membuka project lama atau memberi nama tetap. Source dengan hash/identity/quality cocok dapat digunakan ulang. Project yang sudah berisi sumber berbeda, metadata tidak cocok, atau aset hilang akan ditolak; pilih project baru agar artefak lama tidak tercampur.

**Hasil probe contoh RedNote, 24 Oktober 2026:** yt-dlp terpasang `2026.06.21.235142` belum mengenali URL `rednote.com` langsung. Alias `xiaohongshu.com` dengan ID sama menghasilkan extractor error tanpa metadata. Probe hanya meminta metadata, tanpa cookie/login, instalasi, atau download video. Adapter sudah tersedia, tetapi keberhasilan download contoh tersebut belum terverifikasi. Jika platform menolak akses di Kaggle, gunakan file lokal yang Anda berhak gunakan.

## Word timing dan dependency opsional

Edge menangkap `WordBoundary` dari proses synthesis yang sama dengan audionya. Metadata disimpan setelah validasi urutan, durasi, dan cakupan teks. Dependency Edge dideklarasikan `>=7.3.1,<8` untuk API boundary yang dipakai.

Piper tetap menggunakan engine WAV existing. Untuk subtitle animasi, audio finalnya dapat disejajarkan dengan teks narasi menggunakan extra opsional:

```bash
python -m pip install -e '.[alignment]'
dubbingstory run --input video.mp4 --project cerita_piper --mode full --lang id --engine piper --ratio source --subtitle-style word --word-alignment whisperx --burn-subtitles
```

WhisperX `>=3.8.6,<4` dipilih karena dukungan alignment Indonesia dan perbaikan opsi `ignore` untuk interpolasi. Extra ini membawa dependency/model tambahan; jalur Edge tidak memerlukannya. Pengunduhan model alignment terjadi di lingkungan eksekusi saat pertama dipakai. Pilih `subtitle.alignment_device: cuda` untuk GPU Kaggle yang kompatibel.

`--word-alignment auto` memakai boundary provider yang valid terlebih dahulu, lalu mencoba WhisperX. `edge` memerlukan boundary Edge dan tidak menjalankan fallback. `whisperx` memaksa alignment terhadap teks yang sudah ditetapkan. Tidak ada pembagian waktu rata-rata antar kata.

Angka/singkatan yang dieja berbeda oleh TTS atau kata yang tidak dapat disejajarkan akan menghasilkan pesan koreksi. Tulis bentuk yang diucapkan dalam naskah, lalu ulangi `dub`. Jangan memperbaiki JSON timing secara manual untuk menebak waktu. Kata ber-score rendah atau hilang tidak dianggap sebagai caption yang valid.

## Konfigurasi caption dan audio

Contoh override `native.yaml`:

```yaml
render:
  ratios: [source]
  burn_subtitles: true
  audio_strategy: dynamic_duck
  original_volume: 0.15
  narration_volume: 1.0
  duck_threshold: 0.025
  duck_ratio: 8.0
  duck_attack_ms: 20.0
  duck_release_ms: 350.0
tts:
  narration_fit: rewrite
  fit_retries: 2
  min_gap: 0.20
subtitle:
  style: karaoke
  alignment: auto
  alignment_device: cpu
  font_name: DejaVu Sans
  font_path: /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
  font_size: 64
  font_color: '#FFFFFF'
  active_color: '#FFE600'
  outline_color: '#000000'
  outline: 3
  position: bottom
  animation: pop
  safe_margin_x: 0.08
  safe_margin_y: 0.18
  max_words: 5
  pop_scale: 1.12
```

```bash
dubbingstory --config native.yaml run --input video.mp4 --project cerita_native --lang id
```

`--config` adalah opsi global, diletakkan sebelum subcommand. `font_size` memakai referensi lebar 1080 dan diskalakan ke lebar tampilan output. `font_path` harus menunjuk font yang tersedia pada lingkungan tersebut; font untuk pengukuran layout dan libass harus sama. `font_name` digunakan untuk menemukan font ketika path tidak diisi. Animasi tersedia: `none`, `pop`, `fade`, `scale`; posisi: `top`, `center`, `bottom`.

Pengaturan font di atas berlaku pada subtitle animasi. Style `classic` mempertahankan default sebelumnya (Arial 24, putih, outline hitam 2); override klasik dapat memakai `subtitle.classic_font_name`, `classic_font_size`, `classic_font_color`, `classic_outline_color`, `classic_outline`, `classic_font_path`, dan `classic_position` di bagian `subtitle`.

Safe margins adalah fraksi ukuran tampilan per sisi dan mencakup ukuran animasi maksimum. Nilai contoh merupakan preset, bukan jaminan terhadap seluruh overlay UI platform. Review hasil visual pada target platform.

`original_volume` kini mengatur volume dasar audio asli pada `dynamic_duck` juga. Nilai nol benar-benar mematikan audio asli. Ducking menurunkan audio asli saat narator bersuara dan memulihkannya saat jeda. Video tanpa track audio tetap dapat diproses; audio yang pendek dipad agar video tidak berakhir sebelum waktunya.

## File output dan render ulang

Artefak utama full-mode:

```text
source.mp4                          # Nama legacy; container aktual diperiksa FFprobe
video_metadata.json                 # Identitas sumber, hash, geometri/timeline aktual
ingest_manifest.json
scripts/script_id.txt
scripts/script_id.json              # Teks final setelah fitting
scripts/script_id.srt               # Selalu dipertahankan
scripts/script_id.words.json        # Kata, interval, provenance, hash teks/audio
scripts/script_id.ass               # Salinan layout ASS terakhir
scripts/script_id.<WxH>.<style>.ass  # Layout terpisah per dimensi/style
audio/audio_id.wav
audio/audio_id.manifest.json         # Status, placements, durasi, hash
final_id_source.mp4
render_manifest.json
```

Summary memakai prefix `summary_script_` dan `summary_audio_`, dengan timeline hasil kompilasinya sendiri. Timing kata di artefak sudah berada pada timeline final.

Ganti style tanpa memanggil TTS ulang:

```bash
dubbingstory render --project cerita_native --ratio source --subtitle-style word --burn-subtitles
```

Hash audio, teks, dan sumber diperiksa sebelum strict render. Mengubah salah satunya memerlukan regenerasi yang sesuai. Mengubah style saja memakai word timing yang sama. SRT/JSON kata tidak dihapus saat memilih render klasik.

Project lama tanpa timing kata atau manifest lengkap tetap dapat dirender melalui jalur klasik. Untuk memakai fitur native/animasi, jalankan `dub` kembali, misalnya:

```bash
dubbingstory dub --project proyek_lama --engine edge --narration-fit rewrite --subtitle-style karaoke
dubbingstory render --project proyek_lama --ratio source --subtitle-style karaoke --burn-subtitles
```

Structured script dan video sumber harus tersedia. Tidak ada alignment perkiraan otomatis terhadap project yang tidak lengkap.

## Pengujian di Kaggle

Perubahan ini belum diuji dengan menjalankan tests, TTS, alignment, atau rendering pada komputer development, sesuai permintaan. Perintah berikut **untuk Anda jalankan di Kaggle**, dari root checkout repository yang sudah memuat perubahan ini.

```bash
python -m pip install -e '.[test]'
ffmpeg -hide_banner -version
ffmpeg -hide_banner -filters
python -m pytest -q tests
```

Pastikan filter `subtitles`/libass tersedia. Tests baru memakai provider palsu/fixture; tests FFmpeg memakai media sintetis dan berjalan bila FFmpeg tersedia. Tidak ada test suite yang sengaja melakukan download platform atau TTS cloud. Model nyata dan live provider diuji terpisah. Atur akses Gemini melalui mekanisme Secrets notebook existing sebelum menjalankan pipeline sebenarnya; jangan mencetak nilai key ke output notebook.

Setelah suite lulus, jalankan contoh native di atas pada local video pendek, lalu periksa:

1. Frame awal/akhir, urutan, komposisi, rotation, dan durasi sama dengan sumber. Uji 720×1280, 1080×1920, video ponsel berotasi, dan VFR.
2. Tidak ada ucapan terpotong; overflow yang tidak dapat diperbaiki harus gagal, bukan menghasilkan audio parsial yang dianggap lengkap.
3. Word dan karaoke mengikuti ucapan, termasuk jeda, angka yang dieja, dan perpindahan beat. Target evaluasi manual awal: median error onset ≤80 ms dan p95 ≤150 ms pada sampel beranotasi; ini bukan hasil yang sudah terukur.
4. Caption berada di safe area termasuk ketika pop membesar. Uji font, warna, outline, posisi, dan path dengan spasi.
5. Audio original/ducking bekerja untuk sumber dengan audio, tanpa audio, audio pendek, serta volume nol.
6. Render ulang dengan style berbeda tidak memanggil TTS; perubahan teks/audio ditolak bila timing belum diperbarui.
7. Jalur klasik YouTube/local, full/summary, Edge/Piper tetap dapat digunakan. Uji provider live hanya dengan video yang Anda berhak gunakan.

Simpan log pytest dan contoh output gagal jika ada; keberhasilan review statis tidak menggantikan validasi runtime dan review audiovisual ini.

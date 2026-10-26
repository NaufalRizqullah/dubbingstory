# Render dua layer pada canvas 9:16

Preset `layered` membuat canvas 1080×1920 berisi dua salinan visual dari frame yang sama:

- **Foreground:** video tajam, berwarna, di tengah; default zoom 25.
- **Background:** video di belakang yang memenuhi canvas, dengan blur dan grayscale; default zoom 100, blur 50, grayscale 1.0.

Hanya jalur video yang dibagi menjadi dua. Audio tidak digandakan. Pada pipeline dubbing, mixing/ducking yang sudah ada tetap dipakai sekali. Caption baru dipasang setelah komposisi dua layer, sehingga tidak ikut terkena blur atau muncul dua kali.

## Arti zoom dan crop

Skala zoom mengikuti rentang **0–100**, sesuai kesepakatan penggunaan:

| Nilai | Perilaku |
| --- | --- |
| 0 | Seluruh frame muat di dalam canvas, tanpa memotong isi frame. |
| 25 | Seperempat perpindahan ukuran dari kondisi muat utuh menuju penuh. |
| 100 | Frame memenuhi canvas; bagian di luar canvas terpotong di tengah. |

Secara matematis, `scale = fit + (fill - fit) × zoom / 100`. `fit` adalah skala terkecil agar seluruh frame muat; `fill` adalah skala terbesar agar canvas tertutup. Video tidak ditarik menjadi kotak: aspect ratio isinya tetap, lalu bagian yang melewati batas canvas di-crop secara simetris. Tidak ada deteksi wajah atau perpindahan titik fokus otomatis.

Untuk sumber 1920×1080 pada canvas 1080×1920, zoom foreground 25 menghasilkan ukuran layer sekitar **1664×936**, dengan bagian tengah yang terlihat **1080×936**. Zoom background 100 menghasilkan layer sekitar **3414×1920**, lalu dipotong di tengah menjadi 1080×1920. Pembulatan ke piksel genap dapat menghasilkan selisih kecil. Ini mendekati bentuk kotak pada contoh screenshot, bukan rasio 1:1 yang dipaksakan. Nilai zoom sekitar 36 menghasilkan tinggi foreground mendekati 1080 pada sumber 16:9.

Background di bawah zoom 100 dapat menyisakan area hitam. Untuk menutup seluruh canvas, gunakan default 100. Foreground zoom 100 menutup seluruh background sehingga efek background tidak terlihat.

## Mengolah MP4 yang sudah jadi

Perintah berikut hanya menjalankan tahap komposisi/render, tanpa analisis visual, penulisan naskah, atau TTS:

```bash
python main.py reframe \
  --input final_summary_id_16x9.mp4 \
  --output final_summary_id_vertical.mp4 \
  --vertical-layout layered \
  --foreground-zoom 25 \
  --background-zoom 100 \
  --background-blur 50 \
  --background-grayscale 1.0
```

Gunakan path sumber sebenarnya dan nama output baru. `reframe` menolak input dan output yang sama maupun output yang sudah ada. Canvas `reframe` adalah 9:16 secara default; `--ratio 9:16` boleh ditulis secara eksplisit.

Audio export existing dipakai sekali, dari track audio pertama. Codec audio yang kompatibel disalin; codec lain dikonversi ke AAC. Video tanpa audio tetap bisa diolah. Timing dan panjang audio existing dipertahankan, termasuk audio tail yang sudah lebih panjang daripada video; `reframe` tidak menambahkan trim, padding, gain, atau ducking. Pengaturan TTS, audio mix, atau burn subtitle dari config tidak menjalankan synthesis/mixing/burn baru pada `reframe`. Subtitle yang **sudah menyatu dengan piksel video** akan ikut terduplikasi dan di-crop bersama frame; untuk caption bersih di atas kedua layer, gunakan render pipeline dari source dan subtitle terpisah.

Pengaturan dan hasil reframe dicatat dalam sidecar `<output>.render.json`, misalnya `final_summary_id_vertical.mp4.render.json`.

## Memakai preset pada pipeline full atau summary

```bash
# Menghasilkan dua output: 16:9 existing dan 9:16 layered.
python main.py run --input video.mp4 --project cerita \
  --mode summary --summary-duration 180 --lang id --engine edge \
  --ratio 16:9 9:16 --vertical-layout layered

# Full video; komposisi berubah, urutan frame dan timeline tetap.
python main.py run --input video.mp4 --project cerita_full \
  --mode full --lang id --engine edge \
  --ratio 9:16 --vertical-layout layered

# Render ulang project full yang sudah memiliki source, audio, dan subtitle.
python main.py render --project cerita_full \
  --ratio 9:16 --vertical-layout layered --background-grayscale 0.5
```

Preset hanya berlaku pada output **9:16**. Dalam permintaan multi-ratio, output 16:9/source/1:1/4:3 tetap mengikuti jalur existing. Memilih `layered` tanpa menyertakan 9:16 menghasilkan error agar pilihan layout tidak diabaikan diam-diam. Gunakan `--vertical-layout pad` untuk kembali ke layout black bar existing.

Mode summary tetap memilih dan memotong adegan sesuai pipeline summary. Preset ini hanya mengatur susunan visual setelah kompilasi selesai. `render --project` menggunakan artefak full secara default; `render --project ... --mode summary` menggunakan kompilasi dan audio summary yang tersimpan. Untuk MP4 summary yang sudah diekspor, gunakan `reframe`.

Output pipeline tetap memakai nama existing, misalnya `final_id_9x16.mp4` atau `final_summary_id_9x16.mp4`. Rerender project dapat mengganti hasil render dengan nama yang sama; simpan salinan lebih dahulu jika ingin membandingkan dengan hasil lama. `reframe` menyediakan path output eksplisit yang harus baru.

## Flag dan konfigurasi

Semua flag berikut tersedia pada `run`, `render`, dan `reframe`. Jika flag tidak diisi, nilai config dipakai.

| Flag | Config di bagian `render` | Default | Rentang |
| --- | --- | --- | --- |
| `--vertical-layout` | `vertical_layout` | `pad` | `pad`, `layered` |
| `--foreground-zoom` | `foreground_zoom` | 25 | 0–100 |
| `--background-zoom` | `background_zoom` | 100 | 0–100 |
| `--background-blur` | `background_blur` | 50 | 1–100 |
| `--background-grayscale` | `background_grayscale` | 1.0 | 0–1 |

Grayscale 0 mempertahankan warna, 0.5 mengurangi saturasi setengah, dan 1 membuat background sepenuhnya abu-abu. Foreground tidak terkena blur/grayscale. Nilai blur adalah parameter `sigma` dalam piksel untuk FFmpeg `gblur` dengan `steps=2`; angka yang sama tidak menjamin radius yang identik dengan editor lain.

```yaml
render:
  ratios: ["9:16"]
  vertical_layout: layered
  foreground_zoom: 25
  background_zoom: 100
  background_blur: 50
  background_grayscale: 1.0
```

```bash
python main.py --config vertical.yaml run --input video.mp4 --lang id
```

Preset membutuhkan encoding video karena frame dikomposisikan ulang. Pipeline menggabungkan komposisi dan burn subtitle dalam tahap encode yang sama. Tidak ada filter FPS, perubahan kecepatan, atau pemotongan waktu yang ditambahkan oleh preset. Sumber HDR tetap mengikuti pembatasan color management renderer: gunakan sumber SDR untuk encode sampai kebijakan HDR tersedia. Tidak ada dependency Python baru khusus preset ini; FFmpeg harus menyediakan filter `split`, `scale`, `crop`, `pad`, `gblur`, `hue`, dan `overlay`.

Renderer memakai `-vsync 0` saat encode agar kompatibel dengan build FFmpeg lama
di Kaggle yang belum mengenal opsi `-fps_mode`. Opsi ini tidak mengubah isi
timeline atau menambahkan frame sintetis.

## Penyesuaian contoh notebook Kaggle

Pada contoh `mereska-dubbingstory-yt.py`, ubah pengaturan `RATIO` dan tambahkan variabel berikut:

```python
RATIO = "9:16"
VERTICAL_LAYOUT = "layered"
FOREGROUND_ZOOM = 25
BACKGROUND_ZOOM = 100
BACKGROUND_BLUR = 50
BACKGROUND_GRAYSCALE = 1.0
```

Setelah list `cmd = [...]` untuk `main.py run` dibuat, sebelum `subprocess.Popen(...)`, tambahkan:

```python
cmd.extend([
    "--vertical-layout", VERTICAL_LAYOUT,
    "--foreground-zoom", str(FOREGROUND_ZOOM),
    "--background-zoom", str(BACKGROUND_ZOOM),
    "--background-blur", str(BACKGROUND_BLUR),
    "--background-grayscale", str(BACKGROUND_GRAYSCALE),
])
```

Untuk kedua output sekaligus, ganti elemen `"--ratio", RATIO,` dalam list `cmd` menjadi `"--ratio", "16:9", "9:16",`. Pilihan mode summary, model vision, bahasa, dan suara dapat tetap memakai nilai notebook Anda. Pastikan checkout repository di Kaggle sudah memuat perubahan ini sebelum menjalankan cell pipeline.

Jika MP4 16:9 sudah tersedia, cukup jalankan cell terpisah tanpa menjalankan ulang cell vision/TTS:

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "main.py", "reframe",
    "--input", "/kaggle/working/final_summary_id_16x9.mp4",  # sesuaikan path
    "--output", "/kaggle/working/final_summary_id_layered.mp4",
    "--vertical-layout", "layered",
    "--foreground-zoom", "25",
    "--background-zoom", "100",
    "--background-blur", "50",
    "--background-grayscale", "1.0",
], check=True)
```

## Validasi di Kaggle

### Memulihkan render summary tanpa mengulang AI/TTS

Jika proses sudah menyimpan `summary_source.mp4`, `audio/summary_audio_id.wav`, dan `scripts/summary_script_id.srt`, tetapi kedua MP4 final gagal dibuat, jalankan dari root repository Kaggle yang sudah memuat perbaikan:

```bash
python main.py render --project my_dubbing_project --mode summary \
  --ratio 16:9 9:16 --vertical-layout layered \
  --foreground-zoom 25 --background-zoom 100 \
  --background-blur 50 --background-grayscale 1.0
```

Untuk cell Python notebook:

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "main.py", "render",
    "--project", "my_dubbing_project", "--mode", "summary",
    "--ratio", "16:9", "9:16", "--vertical-layout", "layered",
    "--foreground-zoom", "25", "--background-zoom", "100",
    "--background-blur", "50", "--background-grayscale", "1.0",
], check=True)
```

Perintah ini tidak menjalankan analisis, Gemini, vLLM, scene selection, atau TTS. Artefak summary harus masih tersedia di direktori project (default `outputs/my_dubbing_project`); gunakan config yang sama jika direktori output diubah. Output tetap `final_summary_id_16x9.mp4` dan `final_summary_id_9x16.mp4`. Status tiap rasio disimpan dalam `summary_render_manifest.json`; kegagalan salah satu rasio menghasilkan exit nonzero setelah rasio lain dicoba. Perintah tidak memakai artefak full sebagai pengganti artefak summary yang hilang.

Pada log kegagalan 26 Oktober 2026, kedua rasio berhenti pada `sidechaincompress` dengan `No channel layout for input 1`. Sumber AAC stereo 44,1 kHz dan narasi PCM mono 24 kHz memerlukan penetapan channel layout/format audio yang jelas sebelum ducking. Perbaikan diterapkan pada audio mixer bersama; konfigurasi canvas notebook tidak perlu diubah untuk kasus ini. Tetap gunakan build yang menyertakan perbaikan tersebut.

Render ulang memakai audio yang sudah tersimpan apa adanya. Jika log TTS melaporkan overflow (contoh tersebut: 2,95 detik), perbaikan filter tidak mengembalikan ucapan yang sudah terpotong pada audio legacy. Periksa bagian akhir narasi; jika terpotong, ringkas dan sintesis ulang narasi melalui workflow summary dengan fitting ketat secara terpisah. Jangan menganggap keberhasilan render berarti masalah durasi narasi sudah terselesaikan.

### Tests dan pemeriksaan visual

Tests dan FFmpeg tidak dijalankan pada komputer development sesuai permintaan. Jalankan tests dari checkout terbaru di Kaggle dan review hasil visual sebelum menganggap preset terverifikasi:

```bash
python -m pytest -q tests

# Aktifkan juga regresi FFmpeg sintetis untuk komposisi dan audio campuran.
DUBBINGSTORY_FFMPEG_TESTS=1 python -m pytest -q tests/test_layered_vertical.py tests/test_audio_mix_compat.py
```

Periksa hasil pada video pendek yang mempunyai detail berwarna dan suara jelas:

1. Output 1080×1920, foreground tajam dan berwarna di tengah, background memenuhi canvas pada zoom 100 dan menjadi blur/abu-abu.
2. Zoom 0 menampilkan seluruh frame; zoom 25 memperbesar bagian tengah; zoom 100 memenuhi canvas. Kedua layer bergerak pada frame yang sama.
3. Grayscale 0, 0.5, 1 dan blur 1, 50, 100 mengubah background saja.
4. Hanya satu suara terdengar, tanpa echo/volume ganda; sumber tanpa audio berhasil. Untuk `reframe`, audio narasi hasil export tetap dipertahankan.
5. Durasi dan frame awal/akhir tetap; ulangi pada VFR dan video ponsel dengan rotation metadata.
6. Caption pipeline tetap di atas kedua layer dan berada di safe area. Caption yang sudah di-burn pada input `reframe` mengikuti crop seperti piksel sumber lainnya.
7. Tanpa flag `layered`, output existing tidak berubah; dalam multi-ratio, efek hanya berlaku pada 9:16.

Referensi filter: [FFmpeg overlay](https://ffmpeg.org/ffmpeg-filters.html#overlay), [Gaussian blur](https://ffmpeg.org/ffmpeg-filters.html#gblur), dan [hue/saturation](https://ffmpeg.org/ffmpeg-filters.html#hue).

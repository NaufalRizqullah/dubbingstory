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

### RedNote: domain asli harus dipertahankan

Adapter sekarang mempertahankan `rednote.com` untuk URL RedNote dan `xiaohongshu.com` untuk URL Xiaohongshu. Keduanya tetap memakai provider identity `xiaohongshu:<id>` agar source cache lama dengan hash/quality yang cocok dapat dipakai kembali. Domain pada `canonical_url` mengikuti sumber publik yang digunakan; share query tetap hanya di memori extractor.

**Temuan 28 Oktober 2026 untuk item `6ad206ee000000001400240a`:** URL RedNote asli mengembalikan halaman note dan tiga format video, sedangkan URL dengan ID sama yang ditulis ulang ke Xiaohongshu mengarah ke halaman error HTTP 200 tanpa data note/video. Karena itu, respons HTTP 200 atau pesan `no_video` saja tidak cukup untuk menyimpulkan bahwa post telah dihapus atau hanya berisi gambar. Probe historis di atas memakai item berbeda; kegagalan alias juga tidak membuktikan sumber RedNote aslinya gagal.

`dubbingstory/ingest/rednote.py` memperluas URL matching melalui subclass `RedNoteIE`; seluruh parsing tetap memakai [extractor XiaoHongShu upstream yt-dlp](https://github.com/yt-dlp/yt-dlp/blob/master/yt_dlp/extractor/xiaohongshu.py). Tidak ada login, cookies, browser impersonation, atau pergantian otomatis ke domain lain saat akses ditolak. URL Xiaohongshu tetap memakai extractor upstream seperti sebelumnya.

Metadata dengan selector `max` pada adapter yang sudah diperbaiki berhasil menemukan tiga format, termasuk 720x1280 dengan durasi 22,167 detik. Probe lokal hanya memakai yt-dlp existing `2026.06.21.235142`, tanpa download file video atau menjalankan pipeline. Log Kaggle yang dilaporkan memakai `2026.10.26`; download/render lengkap pada runtime itu masih perlu diuji pengguna.

Untuk memakai perbaikan di Kaggle:

1. Gunakan kode repository yang sudah memuat `dubbingstory/ingest/rednote.py` dan registrasinya di `ingest/youtube.py`, serta notebook terbaru. Notebook saja tidak memperbarui checkout Git yang sudah ada.
2. Pertahankan `VIDEO_INPUT` dengan URL RedNote asli. Setelah normalisasi, log `Source` harus tetap menampilkan `www.rednote.com`.
3. Jika checkout lama masih dipakai, perbarui secara sadar dari perubahan terbaru atau pilih checkout baru. Notebook tidak melakukan pull/reset otomatis dan sekarang memeriksa keberadaan adapter sebelum setup.
4. Jalankan ulang ingest. Jika file `source.mp4` hasil run sebelumnya masih ada dengan metadata/hash/quality cocok, gunakan `PROJECT_NAME` dan `OUTPUT_ROOT` yang sama agar cache dipakai. Jangan menghapus hasil lama hanya untuk mencoba downloader ulang.
5. Jika versi terbaru tetap ditolak platform, gunakan file lokal yang Anda berhak gunakan; jangan mengubah cookies atau menambahkan workaround akses.

Regresi offline terkait ada pada `tests/test_multisource_ingest.py` dan `tests/test_social_notebook.py`; tests belum dijalankan di komputer development.

### Download YouTube di Kaggle

Log notebook 27 Oktober 2026 berhenti di `download_video`, sebelum vision, TTS, atau render. Pesan lama `[no_video]` berasal dari klasifikasi kegagalan pemilihan format yt-dlp; pesan itu sendiri tidak membuktikan bahwa link rusak atau hanya berisi gambar. Logger lama membuang peringatan extractor, sehingga log tersebut tidak cukup untuk memastikan apakah client YouTube, JavaScript/EJS, filter format, atau pembatasan akses yang menjadi pemicu akhirnya.

Downloader sekarang membiarkan versi yt-dlp terpasang menentukan client YouTube untuk video dan subtitle. Penguncian ke `android/web` dihapus agar tidak meniadakan pilihan client yang dipelihara upstream. Batas resolusi dan pengecualian AV1 tetap berlaku; `format_unavailable` berarti format sesuai pilihan tersebut tidak dapat diperoleh, bukan bukti bahwa videonya hilang. Diagnostik memakai pesan yang telah ditentukan di kode, tanpa mencetak URL media bertanda tangan, isi cookie, atau credential.

Pada notebook ini, download dilakukan **sebelum** cell instalasi `requirements.txt`. Karena itu, cell setup downloader harus memasang `yt-dlp[default]`, bukan hanya `yt-dlp`. Extra `default` membawa paket EJS yang sesuai; Deno menyediakan runtime JavaScript. Menurut [panduan resmi EJS](https://github.com/yt-dlp/yt-dlp/wiki/EJS), keduanya diperlukan untuk dukungan YouTube, sedangkan pengambilan EJS dari GitHub bergantung pada akses jaringan.

Gunakan [notebook repository](../notebooks/dubbingstory-qwen-local-universal.ipynb) yang sudah memuat perbaikan. Untuk notebook salinan Anda:

1. Pastikan cell clone menggunakan branch/commit yang sudah memuat perbaikan downloader ini. Pembaruan notebook di repository tidak otomatis mengubah salinan notebook Kaggle.
2. Pada cell bertuliskan `Upgrade yt-dlp`, ubah argumen paket menjadi `"yt-dlp[default]"`. Jalankan setup Deno/FFmpeg dan paket ini sebelum cell download. Perubahan ini dijalankan di Kaggle, bukan komputer development.
3. Mulai dengan session/kernel baru setelah memperbarui paket atau source yang sebelumnya sudah di-import. Mengulang `from ... import ...` saja dapat tetap memakai modul lama dari memori. Notebook repository menghentikan setup bila mendeteksi yt-dlp sudah di-import sebelum upgrade.
4. Jalankan cell download, lalu lanjutkan pipeline hanya setelah berhasil. Notebook repository mencetak versi yt-dlp/EJS, ketersediaan Deno/FFmpeg/FFprobe, dan commit repository untuk memudahkan pemeriksaan log berikutnya.

Tidak perlu memakai `render --mode summary` untuk kegagalan ini. Perintah tersebut hanya memulihkan tahap render ketika video summary dan TTS sudah tersedia. Link yang dapat diputar di browser belum memastikan bahwa format yang sama tersedia bagi downloader di lingkungan Kaggle; bila platform menolak akses, jangan melewati proteksinya. Perubahan ini telah ditinjau secara statis; download langsung dan tests belum dijalankan pada komputer development.

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

## Retry Gemini dan fallback teks lokal

Notebook [TikTok/RedNote](../notebooks/mereska-dubbingstory-tiktok-rednote.ipynb) sekarang mengikuti pola source mode: Python kernel Kaggle, dependency dari `pyproject.toml`, dan `main.py` dari checkout. Setup tidak membuat venv atau membangun editable package. vLLM tetap dipin ke `0.27.1`, versi yang berhasil memuat Qwen pada log yang diberikan pengguna; ini bukan klaim bahwa revisi notebook telah diuji. Dependency project dan vLLM diselesaikan bersama, tanpa `--upgrade` massal. `INSTALL_DEPENDENCIES=False` hanya untuk runtime yang sudah memenuhi dependency tersebut.

Log sebelumnya yang diberikan pengguna memperlihatkan ingest RedNote 720x1280 sekitar 22,17 detik serta analisis Qwen berhasil. Penundaan berikutnya terjadi pada 429/503 Gemini saat memendekkan narasi setelah durasi TTS diukur. Log tersebut belum membuktikan final render selesai. Pada laporan berikutnya, ingest tetap berhasil tetapi runtime vLLM berbeda dan gagal saat import; lihat bagian CUDA di bawah.

Pengaturan default aplikasi ada di `configs/default.yaml`. Fallback lokal **nonaktif pada CLI biasa**, dan **aktif pada preset notebook sosial**:

```yaml
llm:
  provider: gemini                 # local untuk langsung memakai Qwen
  gemini:
    max_attempts: 3                # total termasuk panggilan pertama
    timeout_seconds: 30
    budget_seconds: 90
    initial_wait_seconds: 5
    wait_increment_seconds: 5      # jeda 5 lalu 10 detik
  fallback:
    enabled: true
    state_path: /kaggle/working/dubbingstory_social_outputs/contoh/llm_runtime_state.json
    run_id: contoh-run-unik        # notebook membuat UUID baru per eksekusi pipeline
  local:
    base_url: http://127.0.0.1:8000/v1
    model: Qwen/Qwen3-VL-4B-Instruct
    max_attempts: 2
    timeout_seconds: 60
    budget_seconds: 120
    max_tokens: 2048
    context_window: 10240
```

Semua tahap CLI dalam satu pipeline harus memakai `state_path` dan `run_id` yang sama. Setelah kegagalan layanan Gemini menghabiskan jumlah percobaan atau budget, routing disimpan sebagai lokal untuk sisa run. Run baru dengan ID berbeda boleh mencoba Gemini lagi. JSON/bahasa/struktur yang ditolak mencoba lokal untuk tugas itu saja, tanpa menandai layanan Gemini mati. Kesalahan autentikasi atau request yang tidak dapat di-retry langsung gagal. State hanya menyimpan versi skema, ID run, provider, alasan tetap dan jumlah percobaan; tidak berisi key, prompt atau jawaban. File rusak harus diperiksa atau gunakan `state_path` baru. Jangan menjalankan dua pipeline paralel dengan path state yang sama.

`budget_seconds` adalah budget penjadwalan retry per tugas, bukan batas waktu seluruh stage/video. Timeout transport membatasi operasi request; koneksi/read SDK dan requests bukan hard wall-clock deadline. Respons yang tiba setelah budget ditolak, dan tidak ada retry tambahan setelah budget habis. Upload/pemrosesan video pada Gemini Files API, startup/download model, TTS dan render berada di luar budget ini. `FIT_RETRIES` membatasi jumlah revisi naskah berdasarkan pengukuran audio; tiap revisi memiliki budget LLM sendiri. Retry SDK dinonaktifkan agar tidak melipatgandakan tiga percobaan aplikasi.

Local fallback hanya untuk **teks**. Vision tetap memakai provider yang dikonfigurasi. Notebook memakai server Qwen yang sama untuk vision dan penulisan, tetap hidup sampai `dub` selesai, termasuk resume langsung dari `narrate`/`dub`, lalu dihentikan sebelum render dan melalui cleanup jika pipeline gagal. Dengan `TEXT_LLM_PROVIDER="local"` serta vision Qwen, Secret Gemini tidak diperlukan. `START_FROM="render"` tidak memuat model atau mengambil key.

Endpoint lokal harus loopback dan menyediakan `/tokenize` serta `/v1/chat/completions`. Router memeriksa jumlah token prompt yang telah dibangun, batas server, ruang output dan margin sebelum generation. Prompt tidak dipotong lagi oleh router; beberapa pembuat prompt existing masih merangkum/membatasi evidence. Jika konteks tidak cukup, perbesar `MAX_MODEL_LEN` sesuai kemampuan GPU atau kurangi cakupan analisis secara sadar. Output terpotong, bahasa salah, scene/claim tidak valid, dan rewrite yang melebihi target kata ditolak. Setelah kedua provider gagal pada routing baru, pipeline berhenti; tidak membuat narasi template pengganti. Validasi struktur ini tidak menjamin kualitas cerita atau kesetiaan seluruh klaim visual; review hasil tetap diperlukan.

Untuk melanjutkan hasil log pengguna, set nama project existing dan `START_FROM="dub"` jika source, storyboard, naskah dan metadata masih tersedia. Gunakan `OUTPUT_ROOT` yang menunjuk folder artefak tersebut. Jalankan ulang konfigurasi/ingest agar cache diverifikasi; jangan menjalankan analisis dari awal hanya untuk mencoba fitting ulang. Notebook dan **kode repository** harus sama-sama memuat perubahan ini. Mengunggah notebook saja tidak memperbarui remote Git atau checkout Kaggle; preflight akan menolak checkout lama. Pada checkout existing notebook sengaja tidak melakukan pull/reset otomatis.

Landasan API: [Google GenAI 1.28.0](https://github.com/googleapis/python-genai/blob/main/CHANGELOG.md#1280-2025-07-30) menambahkan retry pada level request; [protokol tokenize vLLM 0.27.1](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/entrypoints/serve/tokenize/protocol.py) menyediakan hitungan token dan batas model; [model card Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) mendukung tugas teks di samping vision. Kualitas narasi Indonesia dan performa fallback pada Kaggle tetap harus diukur.

## Konflik CUDA saat startup vLLM

Log yang diterima pada 28 Oktober 2026 menunjukkan ingest RedNote berhasil, lalu `vllm_server.log` berhenti pada import `transformers.audio_utils` → `torchaudio`, sebelum model dimuat. Pesan akhirnya menyebut **PyTorch CUDA 13.0** dan **TorchAudio CUDA 12.8**. Ini konflik build dependency; menambah retry Gemini atau waktu tunggu server tidak memperbaikinya.

Preset notebook kini memasang `vllm==0.27.1`, `torch==2.13.0+cu130`, `torchvision==0.28.0+cu130`, dan `torchaudio==2.11.0+cu130` bersama dependency project, menggunakan indeks wheel resmi PyTorch. Nomor dasar Torch dan TorchAudio memang berbeda: [requirements resmi vLLM 0.27.1](https://github.com/vllm-project/vllm/blob/v0.27.1/requirements/cuda.txt) menetapkan pasangan tersebut, dan [TorchAudio 2.11 memakai stable ABI](https://docs.pytorch.org/audio/stable/installation.html) untuk PyTorch 2.11 atau lebih baru. Wheel cu130 tersedia untuk Python 3.13 Linux x86_64 pada indeks [Torch](https://download.pytorch.org/whl/cu130/torch/), [TorchVision](https://download.pytorch.org/whl/cu130/torchvision/), dan [TorchAudio](https://download.pytorch.org/whl/cu130/torchaudio/). Mengganti `VLLM_VERSION` memerlukan pembaruan preset pasangan paket; versi yang belum dipetakan ditolak, bukan ditebak.

`pip check` hanya memeriksa metadata dependency. Setup juga wajib memvalidasi versi paket terhadap preset lalu mengimpor Torch/TorchAudio/TorchVision, `transformers.audio_utils`, dan modul API server vLLM dalam subprocess baru, dengan batas 180 detik dan log `dubbingstory_social_install.log`. Preflight ini tetap berjalan ketika `INSTALL_DEPENDENCIES=False`, selama tahap membutuhkan model lokal; runtime versi lama atau varian CUDA berbeda tidak lolos. Mode cloud tanpa fallback dan resume render tidak membutuhkan pemeriksaan vLLM. Import yang berhasil belum membuktikan model akan muat dalam VRAM atau inference berhasil.

Untuk memulihkan sesi yang dilaporkan:

1. Gunakan notebook TikTok/RedNote yang diperbarui dan `INSTALL_DEPENDENCIES=True`. Perbaikan CUDA ini berada di cell notebook; checkout tetap harus memenuhi preflight fitur RedNote existing.
2. Jika Torch/TorchAudio/TorchVision/vLLM pernah diimpor langsung ke kernel, restart kernel sebelum mengganti wheel. Pertahankan folder output project. Jalur notebook memakai subprocess sehingga restart tidak dipaksakan untuk kernel yang belum mengimpor library tersebut.
3. Jalankan ulang konfigurasi dan setup. Pastikan preflight import berhasil; jangan lanjut ketika gagal. Jangan menghapus TorchAudio atau menyamakan nomor dasar versinya secara sembarang.
4. Pakai `PROJECT_NAME` dan `OUTPUT_ROOT` yang sama agar `source.mp4` dapat diverifikasi dan digunakan kembali. Bila `segment_manifest.json` masih tersedia, pilih `START_FROM="analyze"`; bila belum, gunakan `segment`. Artefak yang hilang dari sesi Kaggle sebelumnya tetap perlu dibuat kembali.

Regresi notebook memakai subprocess palsu dan tidak memuat model. Tests hanya disiapkan untuk Kaggle; instalasi wheel, startup model dan pipeline revisi ini belum dijalankan pada komputer development.

## Pengujian di Kaggle

### Pemulihan error canvas audio pada tahap dub

Pada log berikutnya yang diterima 28 Oktober 2026, ingest, segment, analisis Qwen dan narrate sudah berhasil. Ketiga beat TTS juga lolos fitting: 5,59 detik mulai 0,00; 5,26 detik mulai 7,27; dan 5,28 detik mulai 14,72. Error `Rendered narration duration differs from the video timeline` baru terjadi setelah penggabungan canvas audio. Log lama tidak menyertakan durasi WAV hasil atau selisihnya, sehingga besar dan penyebab spesifik drift belum dapat dipastikan dari log itu.

Implementasi sebelumnya mengandalkan batas timestamp `-t`, termasuk ketika `loudnorm` dan resampling aktif. Canvas sekarang memakai jumlah sampel `round(durasi_video * 24000)`, timestamp clip yang dimulai dari nol sebelum penempatan, dan endpoint sampel setelah normalisasi/resampling. Ada tambahan silence internal tiga detik sebelum normalisasi untuk menguras buffer filter; bagian ini dibuang dari output. Tidak ada padding setelah normalisasi untuk menyamarkan ucapan yang hilang. Durasi WAV dibaca dari jumlah sampelnya. [Dokumentasi filter FFmpeg](https://github.com/FFmpeg/FFmpeg/blob/n4.4/doc/filters.texi) menjelaskan resampling 192 kHz pada `loudnorm` dinamis serta endpoint berbasis sampel pada `atrim`.

Guard produksi tetap 20 ms dan narasi yang melampaui window tetap ditolak. Error kini mencantumkan `expected`, `actual`, `delta`, serta status normalisasi. Ini bukan alasan untuk menaikkan toleransi atau melewati validasi. Hasil gagal tidak diterbitkan sebagai audio/manifest lengkap; naskah asli tetap dapat dipakai untuk mencoba `dub` lagi.

Untuk kasus ini, perbarui **kode repository Kaggle dan notebook**, lalu jalankan kembali setup/configuration dengan `PROJECT_NAME`/`OUTPUT_ROOT` yang sama dan `START_FROM="dub"`. Notebook menolak checkout yang belum memuat perbaikan canvas; notebook baru saja tidak memperbarui checkout existing. Pastikan `source.mp4`, `storyboard.json`, `scripts/script_id.json` dan `.txt` masih tersedia. Ingest memverifikasi cache; segment/analyze/narrate tidak perlu diulang. Tahap dub masih melakukan TTS dan fitting ulang karena percobaan sebelumnya belum menerbitkan hasil lengkap. Jangan langsung memilih render.

Regresi media berikut bersifat opt-in dan **hanya untuk dijalankan di Kaggle** setelah checkout diperbarui:

```bash
python -m pytest -q tests/test_native_tts_validation.py tests/test_tts_canvas.py tests/test_social_notebook.py
DUBBINGSTORY_FFMPEG_TESTS=1 python -m pytest -q tests/test_tts_canvas.py
```

Tests media menggunakan WAV sintetis, tanpa Edge/Gemini/Qwen. Durasi pecahan detik, jumlah sampel, gap dan audio dekat ujung diperiksa dengan normalisasi aktif maupun nonaktif. Belum ada klaim tests atau render revisi ini lulus di runtime pengguna.

### Preview tahap 7 dan render tanpa subtitle

Log Kaggle berikutnya pada 28 Oktober 2026 menunjukkan tahap pipeline sampai render berhasil: canvas audio `expected=22.166667s`, `actual=22.166667s`, dan `final_id_source.mp4` selesai dibuat. Error `TypeError: stat ... NoneType` terjadi sesudahnya pada `Video(filename=..., embed=True)` di tahap 7. Pada runtime tersebut, constructor IPython memeriksa argumen `data` yang masih `None`. Notebook kini memakai `Video(str(path), embed=True, width=360)`, membuat ZIP terlebih dahulu, dan menangani kegagalan preview per video sebagai pemberitahuan. Bentuk path positional didukung oleh [API Video IPython](https://github.com/ipython/ipython/blob/main/IPython/core/display.py).

Jika MP4 sudah tersedia dan hanya tahap 7 gagal, cukup salin cell **Hasil dan file download** yang diperbarui ke sesi Kaggle yang sama lalu jalankan cell itu. Tidak perlu mengulang setup, analisis, TTS, atau render. Perbaikan ini berada di notebook; tidak memerlukan perubahan renderer atau instalasi dependency.

`BURN_SUBTITLES=False` pada konfigurasi mematikan burn caption untuk render berikutnya. SRT/ASS dan word timing dapat tetap dihasilkan sebagai file terpisah. Untuk MP4 yang sudah dibuat, gunakan cell render ulang dengan `RUN_RENDER_AGAIN=True` dan `RERENDER_BURN_SUBTITLES=False`, lalu jalankan tahap 7 lagi. Cell ini menyimpan pilihan tersebut ke konfigurasi, memakai video sumber dan audio existing, dan mengganti MP4 dengan nama output yang sama. Subtitle yang memang sudah menjadi bagian gambar video sumber tidak dihapus. Jika hanya mengubah nilai konfigurasi tanpa menjalankan render ulang, MP4 lama tetap seperti sebelumnya.

### Suite dan pemeriksaan audiovisual

Perubahan ini belum diuji dengan menjalankan tests, TTS, alignment, atau rendering pada komputer development, sesuai permintaan. Perintah berikut **untuk Anda jalankan di Kaggle**, dari root checkout repository yang sudah memuat perubahan ini.

```bash
python -m pip install 'pytest>=8,<10' 'jsonschema>=4.21'
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

## Analisis visual rapat dan sinkronisasi tindakan

Mode ini ditujukan untuk video proses, pembuatan, dan perbaikan yang sudah diedit. Seluruh video tetap dipertahankan. Pengamatan visual dibuat rapat; naskah tetap berupa cerita yang mengalir, bukan satu kalimat untuk setiap frame.

```bash
python main.py run --input video.mp4 --mode full --ratio source --lang id \
  --style jozzr_social --temporal-analysis --frame-interval 0.5 \
  --action-sync --engine edge --word-alignment auto \
  --subtitle-style karaoke --no-burn-subtitles
```

`--frame-interval 0.5` berarti dua frame per detik. Nilai `1` atau `2` mengurangi pekerjaan vision. Frame dianalisis dalam batch kecil sesuai `vision.temporal.batch_size`, terpisah dari batas `keyframes.max_per_scene` lama. Jika konteks model terlalu kecil, batch dibagi; frame tidak boleh hilang diam-diam. `vision.temporal.max_frames` positif menghentikan pekerjaan yang melewati batas tersebut, sedangkan `0` tidak membatasi total. Gambar/batch yang cocok dengan source, konfigurasi, dan versi schema dapat dipakai ulang saat `analyze` dilanjutkan.

Observasi lengkap disimpan di `visual_timeline.json`, termasuk `per_second` untuk meninjau hasil setiap detik tanpa panggilan LLM tambahan. Detik yang tidak memiliki sampel ditandai `unsampled`; tidak diisi dengan deskripsi rekaan. Event visual menggunakan rentang sampel, bukan klaim bahwa onset tindakan diketahui persis sampai milidetik. Ketepatan timestamp decoder, terutama video VFR, tetap harus diperiksa dengan media nyata di Kaggle.

Perencana menggabungkan bukti tersebut dengan story plan/memory existing. Prompt memilih bukti yang cukup dan mencatat jumlah yang tidak dimasukkan agar tetap muat dalam konteks model lokal; seluruh frame tetap ada pada artifact. Satu frasa tindakan penting per bagian narasi dipilih secara default (`narration.max_action_anchors`, maksimum dua). Model teks hanya memilih ID kejadian dan frasa dari naskahnya; waktu kejadian diambil dari bukti visual. Penulisan ulang durasi mempertahankan frasa itu dan menerima konteks narasi sebelum/sesudahnya.

Setelah TTS, waktu kata sebenarnya digunakan untuk memilih waktu mulai ucapan di dalam jendela video. Sistem tidak mengubah laju video atau memalsukan durasi kata untuk memenuhi anchor. Penempatan tetap tunduk pada urutan, jarak antarucapan, akhir bagian narasi, dan akhir sumber. Beberapa anchor dapat bertentangan dan tidak selalu dapat dipenuhi hanya dengan menggeser satu klip audio.

- `narration.action_sync_tolerance: 0.5`: toleransi terhadap interval visual, dalam detik; ini target konfigurasi, bukan hasil akurasi terukur.
- `--action-sync-policy best_effort`: tempatkan sebaik mungkin dan laporkan frasa yang meleset.
- `--action-sync-policy error`: hentikan sebelum publikasi audio bila anchor masih meleset. Perbaiki naskah/pengelompokan, lalu ulangi `dub`.
- `no_anchors` berarti tidak ada kejadian cukup pasti yang dipilih, bukan semua aksi sudah tersinkron.

Output tambahan:

| Artifact | Isi |
|---|---|
| `visual_timeline.json` | Status, source hash, konfigurasi sampling, seluruh observasi, event dan ringkasan per detik |
| `temporal_frames/`, `temporal_cache/` | Frame sumber dan cache batch yang tervalidasi |
| `narration_plan.json`, `scripts/script_id.json` | Bukti kejadian, ID anchor, frasa narasi dan jendela waktunya |
| `scripts/script_id.words.json` | Timing kata audio final; tetap dibuat saat action sync aktif tanpa caption |
| `audio/audio_id.actions.json` | Waktu frasa yang diucapkan, jarak dari interval visual, status matched/missed dan hash audio/teks |

ZIP hasil notebook menyertakan timeline visual, storyboard, rencana/memory cerita, naskah, timing kata, dan laporan aksi bila tersedia. Laporan `partial` tetap disertakan pada kebijakan `best_effort` agar frasa yang meleset dapat ditinjau; keberhasilan render tidak sama dengan semua anchor cocok.

Untuk migrasi proyek existing, ulangi `analyze` dengan temporal analysis, kemudian `narrate` dan `dub`. Source dan hasil analisis diikat dengan hash ke naskah; hasil yang tidak cocok ditolak saat resume. `START_FROM="dub"` sesudah mengganti engine memakai naskah final yang sama dan membuat ulang audio/timing. Jika ingin menyimpan kedua versi untuk A/B, salin project output ke nama lain terlebih dahulu. Jangan mengedit timing kata untuk memaksa laporan lulus.

Default CLI tetap kompatibel: `vision.temporal.enabled: false` dan `narration.action_sync: false`. Notebook sosial mengaktifkan keduanya. Fitur ini khusus `full`; `summary` memakai alur existing dan menolak flag temporal/action-sync eksplisit.

### Mengganti engine TTS di notebook

Pilih `TTS_ENGINE`, biarkan `VOICE_ID=""` untuk memakai preset `TTS_VOICES` engine tersebut, dan gunakan `WORD_ALIGNMENT="auto"`. Pergantian engine tidak membawa nama voice dari provider sebelumnya. Pengaturan dan credential awal tetap diperlukan:

| Engine | Timing yang digunakan | Persiapan Kaggle |
|---|---|---|
| `edge` | WordBoundary dari sintesis yang sama | Dependency bawaan; tanpa API key |
| `azure` | WordBoundary melalui SDK opsional | Extra `azure`, Kaggle Secrets `AZURE_SPEECH_KEY` dan `AZURE_SPEECH_REGION`; notebook memilih backend SDK |
| `elevenlabs` | Alignment karakter diubah menjadi kata dan divalidasi | `ELEVENLABS_API_KEY` dan ID voice akun Anda; default model `eleven_multilingual_v2` |
| `piper` | Forced alignment pada audio akhir | Model suara Piper dan extra `alignment` |
| `chirp` | Forced alignment pada audio akhir | Extra `chirp` + `alignment`, Google Cloud ADC yang sudah dikonfigurasi |

Extra dipasang hanya ketika cell setup dijalankan di Kaggle. CLI dapat memasangnya dengan `pip install '.[azure]'`, `'.[chirp,alignment]'`, atau `'.[alignment]'` sesuai kebutuhan. Memilih `--word-alignment provider` mewajibkan timing native; `edge` tetap tersedia sebagai pilihan kompatibilitas khusus Edge. `auto` mencoba timing native yang valid, kemudian WhisperX. Jika metadata native Azure/ElevenLabs/Edge tidak lengkap dan extra alignment belum terpasang, proses berhenti dengan instruksi pemulihan: pilih `WORD_ALIGNMENT="whisperx"`, jalankan setup Kaggle untuk memasang extra tersebut, kemudian ulangi `dub`. Tidak ada pembagian waktu kata secara rata.

Backend Azure REST lama tetap tersedia: `tts.azure_metadata_backend: alignment` memakai audio REST dan forced alignment. Backend `auto` memakai SDK bila tersedia dan endpoint mendukungnya. Kemampuan layanan tertentu, misalnya emotional style atau SSML bebas, tidak otomatis didukung setiap suara/adapter. Kualitas suara baru belum dinilai melalui pengujian dengar.

### Penerimaan di Kaggle

Revisi ini diperiksa secara statis; test suite, model, API TTS, dan render tidak dijalankan pada komputer development. Dari checkout yang memuat perubahan, jalankan suite terlebih dahulu:

```bash
python -m pytest -q tests/test_temporal_analysis.py tests/test_action_story.py \
  tests/test_action_alignment.py tests/test_tts_provider_metadata.py \
  tests/test_enhancement_cli.py tests/test_social_notebook.py
python -m pytest -q tests
```

Lalu gunakan satu video pendek dengan tindakan yang mudah dikenali. Periksa status lengkap semua frame; kecocokan penjelasan dengan urutan proses; tidak ada fakta teknis tanpa bukti; tidak ada ucapan bertabrakan/terpotong; dan tindakan terpilih diucapkan selama interval visual atau toleransi yang ditetapkan. Bandingkan hasil dengan subtitle on/off: timeline ucapan harus sama. Uji Edge dahulu, baru A/B engine lain dengan naskah dan tingkat volume yang sama. Catat durasi proses, kegagalan alignment, pelafalan, kesinambungan, dan penilaian dengar; jangan menyimpulkan kualitas vendor hanya dari keberhasilan render.

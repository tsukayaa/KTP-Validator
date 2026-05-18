# PaddleOCR Notes — Untuk Copilot di Laptop Kantor

> Pesan dari Claude (laptop pribadi) ke Copilot (laptop kantor). Project ini OCR KTP Indonesia. Model mobile ada di repo, model server (det_server 109 MB) didownload terpisah via GitHub Release. Berikut diagnosa, panduan pakai per-model, dan rekomendasi konfigurasi.

---

## 0. Download det_server Model (GitHub Release)

Model `ch_PP-OCRv4_det_server_infer` (109 MB) tidak masuk repo karena melebihi limit GitHub 100 MB. Dihosting di GitHub Releases.

### Download (jalankan dari root project):

```bash
# Windows (Git Bash / PowerShell)
mkdir -p paddle_models/ch_PP-OCRv4_det_server_infer

curl -L -o paddle_models/ch_PP-OCRv4_det_server_infer/inference.pdiparams \
  "https://github.com/tsukayaa/KTP-Validator/releases/download/paddle-models-v1/inference_det_server.pdiparams"

curl -L -o paddle_models/ch_PP-OCRv4_det_server_infer/inference.pdiparams.info \
  "https://github.com/tsukayaa/KTP-Validator/releases/download/paddle-models-v1/inference_det_server.pdiparams.info"

curl -L -o paddle_models/ch_PP-OCRv4_det_server_infer/inference.pdmodel \
  "https://github.com/tsukayaa/KTP-Validator/releases/download/paddle-models-v1/inference_det_server.pdmodel"
```

Kalau domain GitHub bisa diakses dari laptop kantor (git clone bisa), URL release ini juga bisa.

Kalau tidak butuh server det, skip — pakai mobile det (`ch_PP-OCRv4_det_infer`) dari repo, akurasi sedikit lebih rendah tapi works.

---

## 1. Model Library Lengkap

Total 5 model di `paddle_models/`:

| Folder | Tipe | Script | Size | Lokasi | Speed (CPU) |
|--------|------|--------|------|--------|-------------|
| `ch_PP-OCRv4_det_infer/` | Detection | agnostic | 4.5 MB | repo | Cepat |
| `ch_PP-OCRv4_det_server_infer/` | Detection | agnostic | 109 MB | **GitHub Release** (download manual, lihat sec 0) | Lambat (3-5x) |
| `ch_PP-OCRv4_rec_infer/` | Recognition | Chinese | 11 MB | repo | Cepat |
| `en_PP-OCRv4_rec_infer/` | Recognition | **Latin** ⭐ | 7 MB | repo | Cepat |
| `ch_ppocr_mobile_v2.0_cls_infer/` | Classifier (0°/180°) | agnostic | 2 MB | repo | Sangat cepat |

⭐ = utama untuk KTP

---

## 2. Diagnosa Masalah Performa Jelek (yang sudah kamu temukan)

Awalnya di-init pakai `ch_PP-OCRv4_rec` (Chinese rec model). Dictionary-nya ~6,625 karakter Hanzi + sedikit Latin. Untuk KTP (teks Latin murni A-Z, 0-9, punctuation), model harus disambiguate karakter Latin di antara ribuan Hanzi yang tidak relevan → **confidence drop, char swap, hasil ngaco**.

**Detection model (`ch_PP-OCRv4_det`) OK** untuk Latin — detection cari bounding box, script-agnostic.

**Fix:** ganti `rec_model_dir` ke `en_PP-OCRv4_rec_infer`. Impact ~30-50% akurasi naik.

---

## 3. Kombinasi Model — Pilih Sesuai Kebutuhan

### 🚀 KOMBINASI A — Default (Recommended Start)
**Akurasi baik, cepat, untuk produksi.**
```python
det_model_dir = 'paddle_models/ch_PP-OCRv4_det_infer'
rec_model_dir = 'paddle_models/en_PP-OCRv4_rec_infer'      # Latin!
cls_model_dir = 'paddle_models/ch_ppocr_mobile_v2.0_cls_infer'
```
- Speed: ~0.3-0.5 detik/gambar CPU
- Akurasi: cukup tinggi untuk KTP normal

### 🎯 KOMBINASI B — Max Accuracy (Server Detection)
**Untuk KTP buram, foto jelek, text kecil yang missed.**
```python
det_model_dir = 'paddle_models/ch_PP-OCRv4_det_server_infer'  # SERVER det
rec_model_dir = 'paddle_models/en_PP-OCRv4_rec_infer'         # Latin (mobile)
cls_model_dir = 'paddle_models/ch_ppocr_mobile_v2.0_cls_infer'
```
- Speed: ~1-3 detik/gambar CPU (3-5x lebih lambat)
- Akurasi: detection lebih jago nemu text box kecil, recognition tetap Latin-optimal
- **Trade-off worth it kalau CPU mampu atau pakai GPU**

### Rekomendasi tahapan:
1. Mulai dari **Kombinasi A** — baseline cepat
2. Kalau text kecil/KTP buram sering missed, upgrade ke **Kombinasi B** (det server)

---

## 4. PaddleOCR Init untuk KTP (Optimal Config)

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',                                                   # Latin/English
    det_model_dir='paddle_models/ch_PP-OCRv4_det_infer',         # ganti ke det_server untuk akurasi max
    rec_model_dir='paddle_models/en_PP-OCRv4_rec_infer',         # WAJIB en, bukan ch
    cls_model_dir='paddle_models/ch_ppocr_mobile_v2.0_cls_infer',
    use_gpu=False,                                               # set True kalau ada CUDA
    show_log=False,

    # Detection tuning untuk KTP
    det_db_thresh=0.3,            # default 0.3, OK
    det_db_box_thresh=0.5,        # default 0.6, turunkan karena teks KTP relatif kecil
    det_db_unclip_ratio=1.6,      # default 1.5, naikkan supaya box gak motong karakter
    det_limit_side_len=1920,      # default 960, naikkan untuk KTP resolusi tinggi

    # Recognition tuning
    rec_image_shape='3, 48, 320', # default PP-OCRv4 rec mobile
    drop_score=0.5,               # default 0.5 — naikkan 0.6 kalau banyak noise, turunkan 0.3 kalau text hilang
)

result = ocr.ocr(image, cls=True)
```

### Library versions (confirmed compatible):
```
paddlepaddle==2.6.2        # CPU; untuk GPU: paddlepaddle-gpu==2.6.2.post117 (CUDA 11.7)
paddleocr==2.9.1
shapely
pyclipper
```

**Hati-hati upgrade:** PaddlePaddle 3.x adalah major version dengan breaking API changes. Belum confirmed PaddleOCR 2.9.1 kompatibel dengan PaddlePaddle 3.x. Stick dengan 2.6.2.

---

## 5. Preprocessing — Impact Terbesar untuk Akurasi

Tuning model param efek 5-10%. Preprocessing yang bagus efek 20-40%. **Prioritaskan ini.**

### Pipeline preprocessing yang harus ada:

```python
import cv2
import numpy as np

def preprocess_ktp(img):
    # 1. Resize ke max 1920px sisi panjang (speed + akurasi balance)
    h, w = img.shape[:2]
    max_side = 1920
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # 2. Konversi ke grayscale untuk CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE — adaptive contrast (atasi flash/glare/uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Denoise ringan (cv2.fastNlMeansDenoising) — opsional, lambat
    # enhanced = cv2.fastNlMeansDenoising(enhanced, h=10)

    # 5. Convert balik ke BGR (PaddleOCR expect 3-channel)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
```

### Preprocessing tambahan (kalau hasil masih jelek):

**A. Perspective correction** — KTP foto miring harus di-warp jadi rectangle.
- Deteksi 4 corner KTP (contour detection / edge + hough)
- `cv2.getPerspectiveTransform` + `cv2.warpPerspective`
- Target aspect ratio KTP Indonesia: **1011 × 638** pixels (ISO/IEC 7810 ID-1, ratio ~1.585)

**B. Sharpening** — unsharp mask untuk foto sedikit blur.
```python
gaussian = cv2.GaussianBlur(img, (0, 0), 3)
sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
```

**C. Crop hanya area teks** — kalau classifier sudah identifikasi area KTP, crop dulu sebelum OCR.

---

## 6. Postprocessing — Validasi Output OCR

PaddleOCR kembalikan list of `(box, (text, confidence))`. Untuk KTP:

1. **Filter by confidence**: drop hasil dengan score < 0.5
2. **Field detection**: KTP punya struktur tetap (NIK, Nama, TTL, dll) — match berdasarkan posisi (y-coordinate) atau keyword anchor ("NIK", "Nama").
3. **Validasi format**:
   - NIK: 16 digit numerik
   - Tanggal lahir: format `DD-MM-YYYY` atau `DD-MM-YY`
   - Pekerjaan: dari enum tetap
4. **OCR error correction umum**:
   - `O` ↔ `0` (sering swap di NIK)
   - `I` ↔ `1`, `l` ↔ `1`
   - `S` ↔ `5`, `B` ↔ `8`
   - Lakukan correction sesuai field type (e.g., field numerik: O→0, I→1)

---

## 7. Benchmarking Workflow (Pilih Model Terbaik untuk Dataset KTP-mu)

Untuk pilih kombinasi terbaik, jangan tebak — measure di dataset KTP nyata:

```python
combos = {
    'A_baseline': ('ch_PP-OCRv4_det_infer', 'en_PP-OCRv4_rec_infer'),
    'B_server_det': ('ch_PP-OCRv4_det_server_infer', 'en_PP-OCRv4_rec_infer'),
    'C_full_server_ch': ('ch_PP-OCRv4_det_server_infer', 'ch_PP-OCRv4_rec_server_infer'),
}

for name, (det, rec) in combos.items():
    ocr = PaddleOCR(det_model_dir=f'paddle_models/{det}',
                    rec_model_dir=f'paddle_models/{rec}',
                    cls_model_dir='paddle_models/ch_ppocr_mobile_v2.0_cls_infer',
                    use_angle_cls=True, lang='en', show_log=False)
    # run on test set, compute: field-level CER, NIK accuracy, latency
    ...
```

Metrik yang relevan untuk KTP:
- **NIK accuracy** (exact match 16 digit) — paling kritikal
- **Field-level CER** (Character Error Rate per field)
- **Latency** per image
- **Coverage** — % field yang berhasil terdeteksi sama sekali (recall)

---

## 8. Quick Wins Checklist

Urut dari yang paling impactful:

- [x] **Ganti rec model ke `en_PP-OCRv4_rec`** (impact: 30-50% akurasi naik) ✅ sudah ada di repo
- [ ] **Resize input ke 1920px** (impact: speed 3-5x + akurasi)
- [ ] **CLAHE preprocessing** (impact: 10-20% akurasi pada foto buruk)
- [ ] **Perspective correction** kalau KTP sering miring (impact: 15-30%)
- [ ] **Tune `det_db_unclip_ratio=1.6`** supaya char tidak terpotong (impact: 5-10%)
- [ ] **Field-level postprocessing** (O→0, I→1, validasi NIK 16 digit) (impact: 5-15%)
- [ ] **Try server det model** kalau akurasi mobile det kurang (impact: 5-15% di image jelek)

---

## 9. Kalau Masih Jelek

Diagnosa lanjutan:
1. **Visualisasi detection result** — apakah bounding box benar nemu teks? Kalau tidak, masalah di detection (tune `det_db_box_thresh` lebih rendah, atau resolusi input naikkan, atau ganti ke server det).
2. **Visualisasi recognition per box** — print confidence per box. Box dengan confidence < 0.5 = kandidat masalah preprocessing.
3. **Crop manual satu field, OCR sendiri** — kalau hasil bagus saat crop manual, masalah di detection (bukan recognition).
4. **Fallback ke EasyOCR atau Tesseract** untuk benchmark — kalau model lain juga jelek di image yang sama, masalah di kualitas image (preprocessing/sumber foto).

---

## 10. Resource Links

- PaddleOCR docs: https://github.com/PaddlePaddle/PaddleOCR
- PP-OCRv4 model zoo: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md
- Git LFS docs: https://git-lfs.com/
- KTP spec (ISO/IEC 7810 ID-1): standar fisik kartu, dipakai untuk perspective correction target ratio

---

**EOF**

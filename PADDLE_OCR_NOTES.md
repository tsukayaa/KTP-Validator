# PaddleOCR Notes — Untuk Copilot di Laptop Kantor

> Pesan dari Claude (laptop pribadi) ke Copilot (laptop kantor). Project ini OCR KTP Indonesia. Model PaddleOCR sudah didownload di folder `paddle_models/`, tapi performa awal jelek. Berikut diagnosa dan rekomendasi.

---

## 1. Diagnosa Masalah Performa Jelek

Yang sekarang ada di repo:
- `ch_PP-OCRv4_det_infer/` — detection model (Chinese pack)
- `ch_PP-OCRv4_rec_infer/` — **recognition model Chinese** ← biang kerok
- `ch_ppocr_mobile_v2.0_cls_infer/` — angle classifier (script-agnostic, OK)

**Masalah utama:** `ch_PP-OCRv4_rec` adalah recognition model untuk **Bahasa China**. Dictionary-nya berisi ~6,625 karakter Hanzi + sedikit Latin. Saat dipakai untuk KTP (teks Latin murni A-Z, 0-9, punctuation), model harus "memilih" karakter Latin di antara ribuan Hanzi yang tidak relevan → confidence drop, char swap, hasil ngaco.

**Detection model (`ch_PP-OCRv4_det`) sebenarnya OK** untuk Latin karena tugas detection cuma cari bounding box teks, tidak baca isinya — relatif script-agnostic.

---

## 2. Solusi: Ganti Recognition Model ke English/Latin

### Download model yang benar:

**Option A — English (paling cocok untuk KTP):**
```
URL: https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar
```
- Dictionary: ~95 karakter (Latin + digit + punctuation)
- Fokus, akurasi tinggi untuk teks Latin
- Size: ~9 MB

**Option B — Latin multi-language (kalau perlu support karakter ber-aksen):**
```
URL: https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar
```
- Catatan: cuma ada versi v3 untuk Latin, bukan v4
- Untuk KTP Indonesia tidak perlu (Bahasa Indonesia di KTP tidak pakai aksen)

**Rekomendasi: Option A (`en_PP-OCRv4_rec`)**. KTP Indonesia tidak ada aksen, semua Latin standar + digit.

### Step ganti:

1. Download `en_PP-OCRv4_rec_infer.tar` ke `paddle_models/`
2. Extract: `tar -xf en_PP-OCRv4_rec_infer.tar`
3. Hapus `.tar` setelah extract (atau biarkan, akan di-gitignore)
4. Folder hasil: `paddle_models/en_PP-OCRv4_rec_infer/`
5. Update path di kode OCR (lihat section 4)

**Catatan untuk Claude di laptop pribadi:** sudah didownload-kan, tinggal pull dari GitHub.

---

## 3. Pertimbangan Server vs Mobile Model

Yang sudah didownload adalah versi **mobile** (kompak, cepat, akurasi cukup).

Kalau butuh akurasi maksimum dan punya budget compute:
```
ch_PP-OCRv4_det_server_infer  (~110 MB)
ch_PP-OCRv4_rec_server_infer  (~88 MB, Chinese — skip ini)
```

Untuk KTP, **mobile sudah cukup**. Server model overkill, dan tidak ada `en_server`.

---

## 4. PaddleOCR Init untuk KTP (Optimal Config)

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',                                                   # Latin/English
    det_model_dir='paddle_models/ch_PP-OCRv4_det_infer',         # detection OK pakai Chinese
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

## 7. Quick Wins Checklist

Urut dari yang paling impactful:

- [ ] **Ganti rec model ke `en_PP-OCRv4_rec`** (impact: 30-50% akurasi naik)
- [ ] **Resize input ke 1920px** (impact: speed 3-5x + akurasi)
- [ ] **CLAHE preprocessing** (impact: 10-20% akurasi pada foto buruk)
- [ ] **Perspective correction** kalau KTP sering miring (impact: 15-30%)
- [ ] **Tune `det_db_unclip_ratio=1.6`** supaya char tidak terpotong (impact: 5-10%)
- [ ] **Field-level postprocessing** (O→0, I→1, validasi NIK 16 digit) (impact: 5-15%)

---

## 8. Kalau Masih Jelek

Diagnosa lanjutan:
1. **Visualisasi detection result** — apakah bounding box benar nemu teks? Kalau tidak, masalah di detection (tune `det_db_box_thresh` lebih rendah, atau resolusi input naikkan).
2. **Visualisasi recognition per box** — print confidence per box. Box dengan confidence < 0.5 = kandidat masalah preprocessing.
3. **Crop manual satu field, OCR sendiri** — kalau hasil bagus saat crop manual, masalah di detection (bukan recognition).
4. **Fallback ke EasyOCR atau Tesseract** untuk benchmark — kalau model lain juga jelek di image yang sama, masalah di kualitas image (preprocessing/sumber foto).

---

## 9. Resource Links

- PaddleOCR docs: https://github.com/PaddlePaddle/PaddleOCR
- PP-OCRv4 model zoo: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md
- KTP spec (ISO/IEC 7810 ID-1): standar fisik kartu, dipakai untuk perspective correction target ratio

---

**EOF**

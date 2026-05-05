"""
Otomatis download 50 positives + 50 negatives dari Kaggle untuk dataset KTP Validator.

Positives : selfie sambil pegang / menampilkan kartu identitas
Negatives : selfie biasa tanpa kartu identitas

Cara pakai:
    python gather_dataset.py

Pastikan file .env ada di folder yang sama dengan isi:
    KAGGLE_TOKEN=KGAT_xxxxxxxxxxxx   ← token baru (mulai KGAT_)
    atau format lama:
    KAGGLE_TOKEN={"username":"xxx","key":"yyy"}
"""

import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

# ── 1. Baca credentials dari .env ────────────────────────────────────────────

env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    print("ERROR: File .env tidak ditemukan.")
    sys.exit(1)

token_raw = None
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("KAGGLE_TOKEN="):
            token_raw = line[len("KAGGLE_TOKEN="):]
            break

if not token_raw:
    print("ERROR: KAGGLE_TOKEN tidak ditemukan di .env")
    sys.exit(1)

kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)
kaggle_json = kaggle_dir / "kaggle.json"

if token_raw.startswith("KGAT_"):
    # Format token baru Kaggle (bearer token) — set langsung via env var
    os.environ["KAGGLE_TOKEN"] = token_raw
    # kaggle library versi baru membaca KAGGLE_TOKEN langsung dari env
    # Tulis juga ke kaggle.json dengan format yang dikenali versi baru
    kaggle_json.write_text(json.dumps({"token": token_raw}))
    kaggle_json.chmod(0o600)
    print(f"[OK] Kaggle token (KGAT_) di-set via environment variable")
else:
    # Format lama: JSON {"username":"...","key":"..."}
    try:
        token = json.loads(token_raw)
        kaggle_username = token["username"]
        kaggle_key      = token["key"]
    except Exception:
        print("ERROR: Format KAGGLE_TOKEN tidak dikenali.")
        print("  Format baru : KAGGLE_TOKEN=KGAT_xxxxxxxxxxxx")
        print("  Format lama : KAGGLE_TOKEN={\"username\":\"...\",\"key\":\"...\"}")
        sys.exit(1)
    kaggle_json.write_text(json.dumps({"username": kaggle_username, "key": kaggle_key}))
    kaggle_json.chmod(0o600)
    print(f"[OK] Kaggle credentials disimpan ke {kaggle_json}")

# ── 2. Import kaggle ──────────────────────────────────────────────────────────

try:
    import kaggle
    kaggle.api.authenticate()
    print("[OK] Kaggle authenticated sebagai:", kaggle_username)
except ImportError:
    print("Library 'kaggle' belum terinstall. Jalankan:")
    print("    pip install kaggle")
    sys.exit(1)
except Exception as e:
    print("ERROR saat autentikasi:", e)
    sys.exit(1)

# ── 3. Konfigurasi ────────────────────────────────────────────────────────────

N_SAMPLES   = 50
TMP_DIR     = Path("./tmp_kaggle")
POS_OUT     = Path("./data/positives")
NEG_OUT     = Path("./data/negatives")
VALID_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

POS_OUT.mkdir(parents=True, exist_ok=True)
NEG_OUT.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── 4. Helper functions ───────────────────────────────────────────────────────

def download_dataset(owner_slug: str, dest: Path) -> Path:
    """Download dan unzip dataset Kaggle ke folder dest."""
    print(f"\n>> Downloading {owner_slug} ...")
    dest.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(owner_slug, path=str(dest), unzip=True, quiet=False)
    print(f"   Selesai → {dest}")
    return dest


def collect_images(folder: Path, limit: int, exclude_keywords: list = None) -> list:
    """Kumpulkan file gambar dari folder secara rekursif, acak, ambil `limit` buah."""
    exclude_keywords = exclude_keywords or []
    all_files = [
        f for f in folder.rglob("*")
        if f.suffix.lower() in VALID_EXT
        and f.is_file()
        and not any(kw.lower() in f.name.lower() or kw.lower() in str(f).lower()
                    for kw in exclude_keywords)
    ]
    random.seed(42)
    random.shuffle(all_files)
    return all_files[:limit]


def copy_images(files: list, dest: Path, prefix: str) -> int:
    """Copy file ke folder dest dengan nama berurutan."""
    copied = 0
    for i, src in enumerate(files, start=1):
        ext = src.suffix.lower()
        dst = dest / f"{prefix}_{i:03d}{ext}"
        shutil.copy2(src, dst)
        copied += 1
    return copied


# ── 5. Download POSITIVES ─────────────────────────────────────────────────────
# Dataset: tapakah68/selfies-id-images-dataset
# Berisi selfie + foto kartu identitas. Kita ambil folder yg mengandung "id" / "card" / "doc"

print("\n" + "="*55)
print("  STEP 1/2 — Download POSITIVES (selfie + kartu)")
print("="*55)

pos_tmp = TMP_DIR / "positives_raw"

try:
    download_dataset("tapakah68/selfies-id-images-dataset", pos_tmp)
except Exception as e:
    print(f"GAGAL download positives: {e}")
    print("Coba alternatif: trainingdatapro/asian-kyc-photo-dataset")
    try:
        download_dataset("trainingdatapro/asian-kyc-photo-dataset", pos_tmp)
    except Exception as e2:
        print(f"GAGAL juga: {e2}")
        sys.exit(1)

# Prioritas: cari folder/file yang namanya mengandung "id", "card", "kyc", "doc"
priority_keywords = ["id", "card", "kyc", "doc", "identity", "national"]
pos_priority = [
    f for f in pos_tmp.rglob("*")
    if f.suffix.lower() in VALID_EXT
    and f.is_file()
    and any(kw in str(f).lower() for kw in priority_keywords)
]

if len(pos_priority) >= N_SAMPLES:
    random.seed(42)
    random.shuffle(pos_priority)
    pos_files = pos_priority[:N_SAMPLES]
    print(f"   Ditemukan {len(pos_priority)} gambar berlabel kartu → ambil {N_SAMPLES}")
else:
    # Fallback: ambil semua gambar yang ada
    print(f"   Hanya {len(pos_priority)} gambar berlabel kartu, ambil semua yang ada...")
    pos_files = collect_images(pos_tmp, N_SAMPLES)

copied_pos = copy_images(pos_files, POS_OUT, "pos")
print(f"[OK] {copied_pos} positives disalin ke {POS_OUT}")


# ── 6. Download NEGATIVES ─────────────────────────────────────────────────────
# Dataset: jigrubhatt/selfieimagedetectiondataset
# Berisi selfie biasa tanpa kartu identitas

print("\n" + "="*55)
print("  STEP 2/2 — Download NEGATIVES (selfie biasa)")
print("="*55)

neg_tmp = TMP_DIR / "negatives_raw"

try:
    download_dataset("jigrubhatt/selfieimagedetectiondataset", neg_tmp)
except Exception as e:
    print(f"GAGAL download negatives: {e}")
    print("Coba alternatif: AxonData selfie images dari HuggingFace ...")
    # Fallback: download dari HuggingFace via requests
    try:
        import urllib.request
        hf_files = [
            "African/African_Male_22/Phone_indoor_01.jpg",
            "African/African_Male_22/Phone_outdoor_01.jpg",
            "African/African_female_23/Phone_indoor_01.jpg",
            "African/African_female_23/Phone_outdoor_01.jpg",
        ]
        base_url = "https://huggingface.co/datasets/AxonData/Selfie_and_Official_ID_Photo_Dataset/resolve/main/"
        neg_tmp.mkdir(parents=True, exist_ok=True)
        for fname in hf_files:
            url = base_url + fname.replace("/", "/")
            out = neg_tmp / Path(fname).name
            urllib.request.urlretrieve(url, out)
        print("   HuggingFace fallback OK")
    except Exception as e2:
        print(f"GAGAL juga: {e2}")
        sys.exit(1)

# Ambil gambar selfie biasa (hindari yang namanya mengandung "id" / "card")
neg_files = collect_images(neg_tmp, N_SAMPLES, exclude_keywords=["id", "card", "doc", "passport"])
if len(neg_files) < N_SAMPLES:
    # Tanpa filter kalau kurang
    neg_files = collect_images(neg_tmp, N_SAMPLES)

copied_neg = copy_images(neg_files, NEG_OUT, "neg")
print(f"[OK] {copied_neg} negatives disalin ke {NEG_OUT}")


# ── 7. Ringkasan ──────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  SELESAI")
print("="*55)
print(f"  Positives : {len(list(POS_OUT.iterdir()))} gambar → {POS_OUT}")
print(f"  Negatives : {len(list(NEG_OUT.iterdir()))} gambar → {NEG_OUT}")
print()
print("  Langkah selanjutnya:")
print("  1. Cek sekilas isi data/positives/ dan data/negatives/")
print("     pastikan gambarnya sesuai (ada kartu / tidak ada kartu)")
print("  2. Jalankan training:")
print("     python run_xgboost.py")
print("     python run_mobilenet.py")
print()

# Bersihkan folder tmp
try:
    shutil.rmtree(TMP_DIR)
    print(f"  Folder sementara {TMP_DIR} dihapus.")
except Exception:
    print(f"  (Folder {TMP_DIR} bisa dihapus manual)")

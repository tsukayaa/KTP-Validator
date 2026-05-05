"""
Otomatis download 50 positives + 50 negatives dari Kaggle untuk dataset KTP Validator.

Positives : selfie sambil pegang / menampilkan kartu identitas
Negatives : selfie biasa tanpa kartu identitas

Cara pakai:
    python gather_dataset.py

Pastikan file .env ada di folder yang sama dengan isi:
    KAGGLE_TOKEN=KGAT_xxxxxxxxxxxx
"""

import io
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

import requests

# ── 1. Baca token dari .env ───────────────────────────────────────────────────

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

# Dukung dua format:
#   Baru : KGAT_xxxxxxxxxxxx  (bearer token)
#   Lama : {"username":"...","key":"..."}
if token_raw.startswith("KGAT_"):
    AUTH_HEADERS = {"Authorization": f"Bearer {token_raw}"}
    print("[OK] Mode token baru (KGAT_) — pakai Bearer auth")
else:
    try:
        creds = json.loads(token_raw)
        AUTH_HEADERS = {}
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"]      = creds["key"]
        # Untuk requests: gunakan HTTP Basic Auth
        _basic = (creds["username"], creds["key"])
        AUTH_HEADERS = {}   # akan di-override di fungsi download
        print(f"[OK] Mode token lama — user: {creds['username']}")
    except Exception:
        print("ERROR: Format KAGGLE_TOKEN tidak dikenali.")
        print("  Baru : KAGGLE_TOKEN=KGAT_xxxxxxxxxxxx")
        print("  Lama : KAGGLE_TOKEN={\"username\":\"...\",\"key\":\"...\"}")
        sys.exit(1)
    _basic = (creds["username"], creds["key"])

def _auth():
    """Return kwargs untuk requests (headers atau auth)."""
    if token_raw.startswith("KGAT_"):
        return {"headers": AUTH_HEADERS}
    return {"auth": _basic}

# ── 2. Tes koneksi ke Kaggle API ─────────────────────────────────────────────

print(">> Tes koneksi ke Kaggle API ...")
resp = requests.get("https://www.kaggle.com/api/v1/datasets", **_auth(), timeout=15)
if resp.status_code == 401:
    print("ERROR: Token ditolak (401). Pastikan KAGGLE_TOKEN di .env benar.")
    sys.exit(1)
elif resp.status_code not in (200, 400):
    print(f"ERROR: Kaggle API merespons {resp.status_code}: {resp.text[:200]}")
    sys.exit(1)
print("[OK] Kaggle API tersambung\n")

# ── 3. Konfigurasi ────────────────────────────────────────────────────────────

N_SAMPLES = 50
TMP_DIR   = Path("./tmp_kaggle")
POS_OUT   = Path("./data/positives")
NEG_OUT   = Path("./data/negatives")
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

POS_OUT.mkdir(parents=True, exist_ok=True)
NEG_OUT.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── 4. Helper: download + unzip ───────────────────────────────────────────────

def download_dataset(owner_slug: str, dest: Path) -> bool:
    """Download dataset Kaggle via REST API, unzip ke dest.

    Returns True on success, False on failure.
    """
    owner, slug = owner_slug.split("/")
    url = f"https://www.kaggle.com/api/v1/datasets/{owner}/{slug}/download"
    print(f">> Downloading {owner_slug} ...")

    resp = requests.get(url, **_auth(), stream=True, timeout=120,
                        allow_redirects=True)
    if resp.status_code == 403:
        print(f"   GAGAL 403 — dataset membutuhkan persetujuan rules di Kaggle.")
        print(f"   Buka https://www.kaggle.com/datasets/{owner_slug} → Accept rules")
        return False
    if resp.status_code != 200:
        print(f"   GAGAL {resp.status_code}: {resp.text[:200]}")
        return False

    # Tampilkan progress download
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        buf.write(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"\r   {pct:.1f}%  ({downloaded/1024/1024:.1f} MB)", end="", flush=True)
    print()

    buf.seek(0)
    dest.mkdir(parents=True, exist_ok=True)

    content_type = resp.headers.get("content-type", "")
    if "zip" in content_type or resp.headers.get("content-disposition", "").endswith(".zip"):
        with zipfile.ZipFile(buf) as zf:
            zf.extractall(dest)
        print(f"   Unzip selesai → {dest}")
    else:
        # Coba unzip tetap, mungkin zip tapi content-type salah
        try:
            with zipfile.ZipFile(buf) as zf:
                zf.extractall(dest)
            print(f"   Unzip selesai → {dest}")
        except zipfile.BadZipFile:
            out_file = dest / f"{slug}.bin"
            out_file.write_bytes(buf.getvalue())
            print(f"   Disimpan sebagai {out_file}")

    return True


def collect_images(folder: Path, limit: int, exclude_kw: list = None) -> list:
    exclude_kw = exclude_kw or []
    files = [
        f for f in folder.rglob("*")
        if f.suffix.lower() in VALID_EXT and f.is_file()
        and not any(kw in str(f).lower() for kw in exclude_kw)
    ]
    random.seed(42)
    random.shuffle(files)
    return files[:limit]


def copy_to(files: list, dest: Path, prefix: str) -> int:
    count = 0
    for i, src in enumerate(files, 1):
        shutil.copy2(src, dest / f"{prefix}_{i:03d}{src.suffix.lower()}")
        count += 1
    return count


# ── 5. POSITIVES ─────────────────────────────────────────────────────────────

print("=" * 55)
print("  STEP 1/2 — Download POSITIVES (selfie + kartu)")
print("=" * 55)

pos_tmp = TMP_DIR / "pos_raw"
ok = download_dataset("tapakah68/selfies-id-images-dataset", pos_tmp)
if not ok:
    print("   Coba dataset alternatif ...")
    ok = download_dataset("trainingdatapro/asian-kyc-photo-dataset", pos_tmp)
if not ok:
    print("GAGAL: tidak bisa download dataset positives.")
    sys.exit(1)

# Prioritaskan file/folder dengan label kartu
card_kw = ["id", "card", "kyc", "doc", "identity", "national", "passport"]
priority = [
    f for f in pos_tmp.rglob("*")
    if f.suffix.lower() in VALID_EXT and f.is_file()
    and any(kw in str(f).lower() for kw in card_kw)
]
if len(priority) >= N_SAMPLES:
    random.seed(42)
    random.shuffle(priority)
    pos_files = priority[:N_SAMPLES]
    print(f"   {len(priority)} gambar berlabel kartu ditemukan → ambil {N_SAMPLES}")
else:
    print(f"   Hanya {len(priority)} berlabel kartu, ambil dari semua ...")
    pos_files = collect_images(pos_tmp, N_SAMPLES)

n = copy_to(pos_files, POS_OUT, "pos")
print(f"[OK] {n} positives → {POS_OUT}\n")

# ── 6. NEGATIVES ─────────────────────────────────────────────────────────────

print("=" * 55)
print("  STEP 2/2 — Download NEGATIVES (selfie biasa)")
print("=" * 55)

neg_tmp = TMP_DIR / "neg_raw"
ok = download_dataset("jigrubhatt/selfieimagedetectiondataset", neg_tmp)
if not ok:
    print("   Coba dataset alternatif ...")
    ok = download_dataset("tapakah68/selfies-id-images-dataset", neg_tmp)

if not ok:
    print("GAGAL: tidak bisa download dataset negatives.")
    sys.exit(1)

neg_files = collect_images(neg_tmp, N_SAMPLES, exclude_kw=["id", "card", "doc", "passport", "national"])
if len(neg_files) < N_SAMPLES:
    neg_files = collect_images(neg_tmp, N_SAMPLES)

n = copy_to(neg_files, NEG_OUT, "neg")
print(f"[OK] {n} negatives → {NEG_OUT}\n")

# ── 7. Ringkasan ──────────────────────────────────────────────────────────────

print("=" * 55)
print("  SELESAI")
print("=" * 55)
print(f"  Positives : {len(list(POS_OUT.iterdir()))} gambar  →  {POS_OUT}")
print(f"  Negatives : {len(list(NEG_OUT.iterdir()))} gambar  →  {NEG_OUT}")
print()
print("  Cek sekilas isi kedua folder sebelum training.")
print("  Lalu jalankan:")
print("    python run_xgboost.py")
print("    python run_mobilenet.py")
print()

try:
    shutil.rmtree(TMP_DIR)
    print(f"  Folder sementara {TMP_DIR} dihapus.")
except Exception:
    print(f"  (Hapus manual folder {TMP_DIR} kalau masih ada)")

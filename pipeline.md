# KTP Validator — FastAPI Pipeline Setup

## Variabel yang Perlu Diisi

```python
# config.py
CM_BASE_URL = "http://<CM_API_URL>"          # URL API Content Management
CM_API_KEY = "<CM_API_KEY>"                  # Auth key CM (kalau ada)

LIGHTON_OCR_URL = "http://<az2-xllode-lighton-ocr-url>"   # URL Lighton OCR
LIGHTON_API_KEY = "<LIGHTON_API_KEY>"        # Auth key Lighton (kalau ada)

PRUGENAI_URL = "http://<PRUGENAI_URL>"       # URL Prugenai validation
PRUGENAI_API_KEY = "<PRUGENAI_API_KEY>"      # Auth key Prugenai (kalau ada)

CLIP_MODEL_PATH = "/workspace/ktp-validator/model_cache"  # Path CLIP di pod
```

---

## Struktur Project

```
ktp-validator/
├── main.py
├── config.py
├── services/
│   ├── cm_client.py        # Step 1-3: ambil file dari CM
│   ├── clip_service.py     # Step 4: cek ada KTP atau tidak
│   ├── lighton_service.py  # Step 5: OCR extract fields
│   └── prugenai_service.py # Step 6: validasi hasil OCR
├── models/
│   └── schemas.py          # Pydantic models
├── requirements.txt
└── Dockerfile
```

---

## Pipeline Flow

```
POST /validate-ktp
    body: { "spaj_number": "..." }
        │
        ▼
[1] CM API: GET list items by spaj_number → dapat item_id
        │
        ▼
[2] CM API: GET image JSON by item_id → dapat base64 image
        │
        ▼
[3] Decode base64 → PIL Image
        │
        ▼
[4] CLIP model (local) → classify: "KTP" or "bukan KTP"
        │
    ┌───┴───┐
   KTP   bukan KTP
    │         │
    ▼         ▼
[5] Lighton  return { "result": "bukan_ktp" }
    OCR API
        │
        ▼
[6] Prugenai validate extracted fields
        │
        ▼
[7] return {
      "result": "ktp",
      "fields": { ... },
      "validation": { ... }
    }
```

---

## Code

### `config.py`

```python
CM_BASE_URL = "http://<CM_API_URL>"
CM_API_KEY = "<CM_API_KEY>"

LIGHTON_OCR_URL = "http://<az2-xllode-lighton-ocr-url>"
LIGHTON_API_KEY = "<LIGHTON_API_KEY>"

PRUGENAI_URL = "http://<PRUGENAI_URL>"
PRUGENAI_API_KEY = "<PRUGENAI_API_KEY>"

CLIP_MODEL_PATH = "/workspace/ktp-validator/model_cache"
```

---

### `models/schemas.py`

```python
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ValidateKTPRequest(BaseModel):
    spaj_number: str

class ValidateKTPResponse(BaseModel):
    result: str  # "ktp" atau "bukan_ktp"
    fields: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
```

---

### `services/cm_client.py`

```python
import httpx
from config import CM_BASE_URL, CM_API_KEY
import base64
from PIL import Image
import io

async def get_item_id_by_spaj(spaj_number: str) -> str:
    """Step 1: Ambil item_id berdasarkan spaj_number"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{CM_BASE_URL}/items",
            params={"spaj_number": spaj_number},
            headers={"Authorization": f"Bearer {CM_API_KEY}"},
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        # Sesuaikan key-nya dengan response CM yang asli
        return data["item_id"]

async def get_image_by_item_id(item_id: str) -> Image.Image:
    """Step 2-3: Ambil image base64 dari CM dan decode ke PIL Image"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{CM_BASE_URL}/items/{item_id}/document",
            params={"type": "IDIA"},
            headers={"Authorization": f"Bearer {CM_API_KEY}"},
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        # Sesuaikan key-nya dengan response CM yang asli
        image_base64 = data["image_base64"]
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
```

---

### `services/clip_service.py`

```python
import torch
import clip
from PIL import Image
from config import CLIP_MODEL_PATH

# Load model sekali waktu startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root=CLIP_MODEL_PATH)

KTP_LABELS = ["an Indonesian ID card KTP", "not an ID card"]

def is_ktp(image: Image.Image) -> bool:
    """Return True kalau CLIP classify sebagai KTP"""
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(KTP_LABELS).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        logits_per_image, _ = model(image_input, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # index 0 = KTP, index 1 = bukan KTP
    return bool(probs[0][0] > probs[0][1])
```

> **Catatan:** Sesuaikan label `KTP_LABELS` dan cara load model dengan implementasi CLIP kamu yang udah ada di `/workspace/ktp-validator`.

---

### `services/lighton_service.py`

```python
import httpx
import base64
import io
from PIL import Image
from config import LIGHTON_OCR_URL, LIGHTON_API_KEY
from typing import Dict, Any

async def extract_ktp_fields(image: Image.Image) -> Dict[str, Any]:
    """Step 5: Kirim image ke Lighton OCR, return extracted fields"""
    # Convert PIL Image ke base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{LIGHTON_OCR_URL}/ocr",   # Sesuaikan endpoint-nya
            json={"image": image_base64},
            headers={"Authorization": f"Bearer {LIGHTON_API_KEY}"},
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()
```

---

### `services/prugenai_service.py`

```python
import httpx
from config import PRUGENAI_URL, PRUGENAI_API_KEY
from typing import Dict, Any

async def validate_fields(ocr_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Step 6: Validasi hasil OCR ke Prugenai"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{PRUGENAI_URL}/validate",  # Sesuaikan endpoint-nya
            json=ocr_fields,
            headers={"Authorization": f"Bearer {PRUGENAI_API_KEY}"},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
```

---

### `main.py`

```python
from fastapi import FastAPI, HTTPException
from models.schemas import ValidateKTPRequest, ValidateKTPResponse
from services.cm_client import get_item_id_by_spaj, get_image_by_item_id
from services.clip_service import is_ktp
from services.lighton_service import extract_ktp_fields
from services.prugenai_service import validate_fields
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KTP Validator", version="0.1.0")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/validate-ktp", response_model=ValidateKTPResponse)
async def validate_ktp(request: ValidateKTPRequest):
    try:
        logger.info(f"Processing spaj_number: {request.spaj_number}")

        # Step 1-2: Ambil image dari CM
        item_id = await get_item_id_by_spaj(request.spaj_number)
        logger.info(f"Got item_id: {item_id}")

        image = await get_image_by_item_id(item_id)
        logger.info("Image fetched and decoded")

        # Step 3: CLIP classification
        ktp_detected = is_ktp(image)
        logger.info(f"CLIP result: {'KTP' if ktp_detected else 'bukan KTP'}")

        if not ktp_detected:
            return ValidateKTPResponse(result="bukan_ktp")

        # Step 4: Lighton OCR
        ocr_fields = await extract_ktp_fields(image)
        logger.info(f"OCR fields: {ocr_fields}")

        # Step 5: Prugenai validation
        validation_result = await validate_fields(ocr_fields)
        logger.info(f"Prugenai result: {validation_result}")

        return ValidateKTPResponse(
            result="ktp",
            fields=ocr_fields,
            validation=validation_result
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e}")
        raise HTTPException(status_code=502, detail=f"External service error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

### `requirements.txt`

```
fastapi
uvicorn[standard]
httpx
pillow
torch
clip @ git+https://github.com/openai/CLIP.git
pydantic
```

> **Catatan:** Kalau CLIP udah keinstall di pod, ga perlu include lagi di requirements.

---

### `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Mount /workspace/ktp-validator dari luar (via volumeMount di K8s)
# CLIP_MODEL_PATH harus pointing ke sana

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Cara Jalanin Lokal (Testing Dulu)

```bash
# Install deps
pip install -r requirements.txt

# Jalanin
uvicorn main:app --reload --port 8000

# Test
curl -X POST http://localhost:8000/validate-ktp \
  -H "Content-Type: application/json" \
  -d '{"spaj_number": "TEST123"}'
```

---

## Yang Harus Kamu Sesuaikan

| Item | Lokasi | Keterangan |
|------|--------|------------|
| CM API endpoint + response key | `cm_client.py` | Sesuaikan dengan payload yang udah dikasih |
| Lighton OCR endpoint + payload format | `lighton_service.py` | Cek dokumentasi/contoh request Lighton |
| Prugenai endpoint + payload format | `prugenai_service.py` | Figur out nanti, bisa mock dulu |
| CLIP load model cara-nya | `clip_service.py` | Sesuaikan dengan script yang udah ada di `/workspace/ktp-validator` |
| Auth header format | Semua service | Bearer token? API key di header custom? |

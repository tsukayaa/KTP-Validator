# OCR Evaluation Framework: PaddleOCR vs LightonOCR2-1-B
## KTP (Indonesian ID Card) Validator Use Case

---

## 0. Context & Scope

- **Task**: Compare PaddleOCR and LightonOCR2-1-B for extracting structured fields from selfie+KTP images
- **Dataset**: ~40–100 images, each paired with a human-verified ground truth JSON
- **Ground truth format**: JSON with `fields` (structured key-value) and `ocr_lines` (raw OCR lines from annotation)
- **Priority**: Accuracy > Speed
- **Constraint**: CPU-only, fully local

### Ground Truth JSON Structure
```json
{
  "source_file": "Foto Selfie dengan KTP_xxx.tiff",
  "verified": true,
  "fields": {
    "Provinsi": "PROVINSI BALI",
    "Kabupaten/Kota": "KABUPATEN BADUNG",
    "NIK": "5103067112680275",
    "Nama": "NINENGAH RUMIASIH",
    "Tempat/Tgl Lahir": "KARANGASEM, 31-12-1968",
    "Jenis Kelamin": "PEREMPUAN",
    "Gol. Darah": "AB",
    "Alamat": "JL. CANGGU GG. TERATAI 1 NO.8 LINGK ANYAR KAJA",
    "RT/RW": "000/000",
    "Kel/Desa": "KEROBOKAN",
    "Kecamatan": "KUTA UTARA",
    "Agama": "HINDU",
    "Status Perkawinan": "KAWIN",
    "Pekerjaan": "MENGURUS RUMAH TANGGA",
    "Kewarganegaraan": "WNI",
    "Berlaku Hingga": "SEUMUR HIDUP"
  }
}
```

### Known OCR Output Characteristics
| OCR Engine | Output Format | Notes |
|---|---|---|
| **LightonOCR2-1-B** | `"Label : Value\nLabel : Value\n..."` | Preserves label-value pairing in single line |
| **PaddleOCR** | Flat list of boxes, labels and values separated | `bbox + text + confidence` per detection |

---

## 1. Evaluation Architecture

Two independent evaluation layers. **Both use the same metrics.** Different extraction strategies per OCR are allowed — but the final metrics must be identical so results are comparable.

```
Image
  │
  ├─► LightonOCR2  ──► raw_text (label:value format)
  │                         │
  └─► PaddleOCR    ──► raw_text (flat token list)
                            │
                    [Layer 1: Raw OCR Quality]
                    CER, WER, Fuzzy Token Recall,
                    Character Precision
                            │
                    [Layer 2: Field Extraction]
                    Field Exact Match, Fuzzy Match,
                    Field Detection Rate, NIK Exact Match
                            │
                    [Layer 3: Speed]
                    Latency per image (end-to-end)
```

---

## 2. Preprocessing: Normalize OCR Output

Before any metric computation, normalize both GT and OCR output using the **same normalization function**. This eliminates trivial differences (case, extra spaces) that are not meaningful for KTP validation.

```python
import re

def normalize_text(text: str) -> str:
    """
    Standard normalization applied to BOTH ground truth and OCR output.
    Must be identical for all OCR engines — no engine-specific normalization.
    """
    text = text.upper().strip()
    text = re.sub(r'\s+', ' ', text)          # collapse multiple spaces
    text = re.sub(r'[^\w\s\/\.\-\,\:]', '', text)  # keep alphanumeric + KTP punctuation
    return text
```

**Important**: Do NOT do engine-specific normalization. If you fix PaddleOCR's output but not LightonOCR's, the comparison is invalid.

---

## 3. Layer 1 — Raw OCR Quality (Layout-Agnostic)

### 3.1 Rationale

OCR engines output boxes in different orders. Concatenating all boxes into a single string and computing CER/WER would be **order-sensitive** and unfairly penalize engines that read right-to-left or column-by-column. Use **order-agnostic** metrics instead.

### 3.2 Build Ground Truth Token Set

Use `fields` values only (not `ocr_lines`) as ground truth for Layer 1, since `fields` is your human-verified source of truth.

```python
def build_gt_token_set(fields: dict) -> set[str]:
    """
    Flatten all GT field values into a set of normalized tokens.
    Order-agnostic: treats OCR output as a bag of words.
    """
    all_text = ' '.join(fields.values())
    normalized = normalize_text(all_text)
    return set(normalized.split())
```

### 3.3 Build OCR Token Set

```python
def build_ocr_token_set(ocr_raw: str) -> set[str]:
    """
    ocr_raw: full raw string output from OCR engine
    (concatenate all detected text with spaces)
    """
    normalized = normalize_text(ocr_raw)
    return set(normalized.split())
```

### 3.4 Metrics

#### CER (Character Error Rate)
Use `jiwer` or `python-Levenshtein`. Apply on **sorted token list** (not raw concatenation) to reduce order bias.

```python
from jiwer import cer, wer

def compute_cer_wer(gt_fields: dict, ocr_raw: str):
    """
    Sort tokens before joining to reduce order-sensitivity.
    This is still imperfect but better than raw concatenation.
    """
    gt_tokens = sorted(build_gt_token_set(gt_fields))
    ocr_tokens = sorted(build_ocr_token_set(ocr_raw))

    gt_str = ' '.join(gt_tokens)
    ocr_str = ' '.join(ocr_tokens)

    return {
        'CER': cer(gt_str, ocr_str),
        'WER': wer(gt_str, ocr_str)
    }
```

> ⚠️ **Caveat**: CER/WER on sorted tokens is still an approximation. Treat as directional signal, not absolute truth. Fuzzy Token Recall (below) is more reliable for this use case.

#### Fuzzy Token Recall (Primary Layer 1 Metric)
For each GT value, check if it appears anywhere in the OCR output with fuzzy matching. Order-agnostic.

```python
from rapidfuzz import fuzz, process

def fuzzy_token_recall(gt_fields: dict, ocr_raw: str, threshold: int = 80) -> dict:
    """
    For each GT field value, check if OCR output contains it (fuzzy).
    Returns per-field result and overall recall score.
    """
    ocr_normalized = normalize_text(ocr_raw)
    results = {}

    for field, gt_value in gt_fields.items():
        gt_norm = normalize_text(gt_value)
        # Use partial_ratio to handle substring matches (value inside larger string)
        score = fuzz.partial_ratio(gt_norm, ocr_normalized)
        results[field] = {
            'gt_value': gt_value,
            'match_score': score,
            'found': score >= threshold
        }

    found_count = sum(1 for r in results.values() if r['found'])
    results['__recall__'] = found_count / len(gt_fields)
    return results
```

#### Character Precision
Penalize hallucination — OCR output contains characters not in GT.

```python
def character_precision(gt_fields: dict, ocr_raw: str) -> float:
    """
    What % of characters in OCR output also appear in GT?
    Low precision = OCR is hallucinating content.
    """
    gt_chars = set(normalize_text(' '.join(gt_fields.values())))
    ocr_chars = set(normalize_text(ocr_raw))

    if not ocr_chars:
        return 0.0

    overlap = gt_chars & ocr_chars
    return len(overlap) / len(ocr_chars)
```

### 3.5 Layer 1 Summary Output per Image

```python
{
    "image": "Foto Selfie dengan KTP_xxx.tiff",
    "ocr_engine": "LightonOCR2",
    "layer1": {
        "CER": 0.12,
        "WER": 0.18,
        "fuzzy_token_recall": 0.87,
        "character_precision": 0.91
    }
}
```

---

## 4. Layer 2 — Field Extraction Quality

### 4.1 Rationale

This layer answers the real business question: **"Can this OCR reliably extract KTP fields for validation?"**

Since LightonOCR and PaddleOCR have different output formats, they need different **extractors** — but the **evaluation metrics must be identical**.

### 4.2 Extractor: LightonOCR2

LightonOCR output format: `"Label : Value\nLabel2 : Value2\n..."`

```python
import re

# KTP field anchors — map known label variations to canonical field names
FIELD_ANCHORS = {
    'Provinsi': ['PROVINSI'],
    'Kabupaten/Kota': ['KABUPATEN', 'KOTA'],
    'NIK': ['NIK'],
    'Nama': ['NAMA'],
    'Tempat/Tgl Lahir': ['TEMPAT', 'TGL LAHIR', 'TTL', 'TEMPAT/TGL'],
    'Jenis Kelamin': ['JENIS KELAMIN', 'JENIS KETAMIN'],  # common OCR typo
    'Gol. Darah': ['GOL DARAH', 'GOL. DARAH', 'DARAH'],
    'Alamat': ['ALAMAT'],
    'RT/RW': ['RT/RW', 'RTARW', 'RT RW'],
    'Kel/Desa': ['KEL/DESA', 'KELURAHAN', 'DESA', 'KEV DESA'],
    'Kecamatan': ['KECAMATAN'],
    'Agama': ['AGAMA'],
    'Status Perkawinan': ['STATUS PERKAWINAN', 'STATUS PERKAWNAN', 'STALUS'],
    'Pekerjaan': ['PEKERJAAN', 'PEKENAANAE'],
    'Kewarganegaraan': ['KEWARGANEGARAAN'],
    'Berlaku Hingga': ['BERLAKU HINGGA', 'BERLAKU HNGGS'],
}

def extract_fields_lighton(ocr_raw: str) -> dict:
    """
    Extract KTP fields from LightonOCR output.
    Format assumption: "Label : Value" per line.
    """
    extracted = {}
    lines = ocr_raw.split('\n')

    for line in lines:
        line_norm = normalize_text(line)

        # Try splitting on ' : ' or ':'
        if ':' in line_norm:
            parts = line_norm.split(':', 1)
            label_part = parts[0].strip()
            value_part = parts[1].strip() if len(parts) > 1 else ''

            # Match label_part to known field anchors
            for canonical_field, anchors in FIELD_ANCHORS.items():
                for anchor in anchors:
                    if anchor in label_part:
                        extracted[canonical_field] = value_part
                        break

    return extracted
```

### 4.3 Extractor: PaddleOCR

PaddleOCR output: flat list of `(bbox, text, confidence)`. Use anchor keyword proximity.

```python
def extract_fields_paddle(paddle_results: list[tuple]) -> dict:
    """
    paddle_results: list of (bbox, text, confidence)
    bbox format: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

    Strategy:
    1. Find box matching a known label anchor
    2. Look for nearest box below or to the right
    3. That box's text = field value
    """
    def get_center(bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (sum(xs)/4, sum(ys)/4)

    def find_nearest_value(label_bbox, all_boxes, direction='below', threshold=50):
        """Find closest box in specified direction."""
        lx, ly = get_center(label_bbox)
        candidates = []

        for bbox, text, conf in all_boxes:
            cx, cy = get_center(bbox)
            if direction == 'below':
                if cy > ly + 5 and abs(cx - lx) < threshold:  # roughly same column
                    candidates.append((cy - ly, text, conf))
            elif direction == 'right':
                if cx > lx + 5 and abs(cy - ly) < 20:  # same row
                    candidates.append((cx - lx, text, conf))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]  # return nearest text
        return None

    extracted = {}
    normalized_boxes = [(bbox, normalize_text(text), conf) for bbox, text, conf in paddle_results]

    for bbox, text, conf in normalized_boxes:
        for canonical_field, anchors in FIELD_ANCHORS.items():
            for anchor in anchors:
                if text.strip() == anchor or text.startswith(anchor):
                    # Try right first (same line), then below
                    value = find_nearest_value(bbox, normalized_boxes, direction='right')
                    if not value:
                        value = find_nearest_value(bbox, normalized_boxes, direction='below')
                    if value:
                        extracted[canonical_field] = value
                    break

    return extracted
```

### 4.4 Field-Level Metrics

Apply these metrics to the `extracted` dict vs `gt_fields` dict, **identically for both OCR engines**.

```python
from rapidfuzz import fuzz

# Fields that require EXACT match (no tolerance)
EXACT_MATCH_FIELDS = {'NIK', 'Gol. Darah', 'RT/RW', 'Jenis Kelamin', 'Kewarganegaraan', 'Berlaku Hingga'}

# Fields that allow fuzzy match (prone to OCR variation)
FUZZY_MATCH_FIELDS = {'Nama', 'Alamat', 'Tempat/Tgl Lahir', 'Pekerjaan', 'Kel/Desa', 'Kecamatan', 'Provinsi', 'Kabupaten/Kota', 'Agama', 'Status Perkawinan'}

FUZZY_THRESHOLD = 85  # minimum fuzzy score to count as match

def evaluate_field_extraction(gt_fields: dict, extracted_fields: dict) -> dict:
    results = {}

    for field, gt_value in gt_fields.items():
        gt_norm = normalize_text(gt_value)
        ocr_value = extracted_fields.get(field)

        if ocr_value is None:
            results[field] = {
                'gt': gt_value,
                'ocr': None,
                'detected': False,
                'exact_match': False,
                'fuzzy_match': False,
                'fuzzy_score': 0
            }
            continue

        ocr_norm = normalize_text(ocr_value)
        fuzzy_score = fuzz.ratio(gt_norm, ocr_norm)
        exact = (gt_norm == ocr_norm)

        if field in EXACT_MATCH_FIELDS:
            match = exact
        else:
            match = fuzzy_score >= FUZZY_THRESHOLD

        results[field] = {
            'gt': gt_value,
            'ocr': ocr_value,
            'detected': True,
            'exact_match': exact,
            'fuzzy_match': match,
            'fuzzy_score': fuzzy_score
        }

    return results


def aggregate_field_metrics(all_results: list[dict]) -> dict:
    """
    all_results: list of per-image field evaluation results
    Returns macro-averaged metrics across all images and fields.
    """
    field_stats = {}

    for image_result in all_results:
        for field, result in image_result.items():
            if field not in field_stats:
                field_stats[field] = {'detected': 0, 'exact': 0, 'fuzzy': 0, 'total': 0}
            field_stats[field]['total'] += 1
            field_stats[field]['detected'] += int(result['detected'])
            field_stats[field]['exact'] += int(result['exact_match'])
            field_stats[field]['fuzzy'] += int(result['fuzzy_match'])

    summary = {}
    for field, stats in field_stats.items():
        n = stats['total']
        summary[field] = {
            'detection_rate': stats['detected'] / n,
            'exact_match_rate': stats['exact'] / n,
            'fuzzy_match_rate': stats['fuzzy'] / n,
        }

    return summary
```

### 4.5 NIK Special Handling

NIK is the most critical field. Evaluate separately with regex as a sanity check.

```python
import re

def nik_eval(gt_fields: dict, ocr_raw: str) -> dict:
    """
    Independent NIK evaluation using regex on raw OCR output.
    Does NOT depend on the field extractor — zero extractor bias.
    """
    gt_nik = gt_fields.get('NIK', '')
    
    # Find all 16-digit sequences in raw OCR output
    candidates = re.findall(r'\b\d{16}\b', ocr_raw.replace(' ', ''))
    
    return {
        'gt_nik': gt_nik,
        'candidates_found': candidates,
        'exact_match': gt_nik in candidates,
        'any_16digit_found': len(candidates) > 0
    }
```

---

## 5. Layer 3 — Speed

Measure **end-to-end latency**: from image path in to extracted fields out. Include warmup.

```python
import time

def benchmark_ocr(ocr_fn, image_paths: list[str], warmup: int = 5) -> dict:
    """
    ocr_fn: callable that takes image_path and returns raw OCR output
    """
    # Warmup — don't include in results
    for path in image_paths[:warmup]:
        _ = ocr_fn(path)

    latencies = []
    for path in image_paths:
        start = time.perf_counter()
        _ = ocr_fn(path)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    return {
        'mean_ms': sum(latencies) / len(latencies),
        'median_ms': sorted(latencies)[len(latencies) // 2],
        'p95_ms': sorted(latencies)[int(len(latencies) * 0.95)],
        'min_ms': min(latencies),
        'max_ms': max(latencies),
    }
```

---

## 6. Full Evaluation Pipeline

```python
import json
import os

def run_evaluation(
    gt_json_dir: str,
    lighton_ocr_fn,   # callable: image_path -> raw_str
    paddle_ocr_fn,    # callable: image_path -> list[(bbox, text, conf)]
    paddle_raw_fn,    # callable: image_path -> raw_str (for Layer 1)
    image_dir: str
) -> dict:

    results = {'LightonOCR2': [], 'PaddleOCR': []}

    for json_file in os.listdir(gt_json_dir):
        if not json_file.endswith('.json'):
            continue

        with open(os.path.join(gt_json_dir, json_file)) as f:
            gt = json.load(f)

        gt_fields = gt['fields']
        image_path = os.path.join(image_dir, gt['source_file'])

        if not os.path.exists(image_path):
            print(f"Warning: image not found: {image_path}")
            continue

        # --- LightonOCR2 ---
        lighton_raw = lighton_ocr_fn(image_path)
        lighton_extracted = extract_fields_lighton(lighton_raw)

        lighton_result = {
            'image': gt['source_file'],
            'layer1': {
                **compute_cer_wer(gt_fields, lighton_raw),
                'fuzzy_token_recall': fuzzy_token_recall(gt_fields, lighton_raw),
                'character_precision': character_precision(gt_fields, lighton_raw),
            },
            'layer2': evaluate_field_extraction(gt_fields, lighton_extracted),
            'nik': nik_eval(gt_fields, lighton_raw),
        }
        results['LightonOCR2'].append(lighton_result)

        # --- PaddleOCR ---
        paddle_detections = paddle_ocr_fn(image_path)   # list[(bbox, text, conf)]
        paddle_raw = paddle_raw_fn(image_path)           # raw string for Layer 1
        paddle_extracted = extract_fields_paddle(paddle_detections)

        paddle_result = {
            'image': gt['source_file'],
            'layer1': {
                **compute_cer_wer(gt_fields, paddle_raw),
                'fuzzy_token_recall': fuzzy_token_recall(gt_fields, paddle_raw),
                'character_precision': character_precision(gt_fields, paddle_raw),
            },
            'layer2': evaluate_field_extraction(gt_fields, paddle_extracted),
            'nik': nik_eval(gt_fields, paddle_raw),
        }
        results['PaddleOCR'].append(paddle_result)

    return results
```

---

## 7. Output & Reporting

### 7.1 Per-Engine Summary Table

Generate this table after aggregation:

| Metric | LightonOCR2 | PaddleOCR |
|--------|-------------|-----------|
| CER (↓) | | |
| WER (↓) | | |
| Fuzzy Token Recall (↑) | | |
| Character Precision (↑) | | |
| NIK Exact Match Rate (↑) | | |
| Overall Field Detection Rate (↑) | | |
| Overall Fuzzy Match Rate (↑) | | |
| Mean Latency ms (↓) | | |
| P95 Latency ms (↓) | | |

### 7.2 Per-Field Breakdown

| Field | Lighton Detection | Lighton Fuzzy Match | Paddle Detection | Paddle Fuzzy Match |
|-------|------------------|---------------------|------------------|--------------------|
| NIK | | | | |
| Nama | | | | |
| Alamat | | | | |
| ... | | | | |

### 7.3 Failure Analysis

For any field where match rate < 80%, manually inspect:
- Is it a normalization issue? (fixable)
- Is it a systematic OCR error? (e.g., always misreads 'NG' as 'N6')
- Is it an extractor issue? (label not in anchor list)

---

## 8. Dependencies

```txt
jiwer>=3.0.3
rapidfuzz>=3.6.1
python-Levenshtein>=0.23.0
```

Install:
```bash
pip install jiwer rapidfuzz python-Levenshtein
```

---

## 9. Known Limitations & Caveats

1. **CER/WER on sorted tokens** is an approximation to reduce order-bias. It is not a perfect measure. Fuzzy Token Recall is more reliable for this use case.

2. **Field anchors in `FIELD_ANCHORS`** need to be extended based on actual OCR errors observed. Both extractors use the same anchor list — update it from real data, not assumptions.

3. **PaddleOCR spatial extractor** uses a fixed `threshold=50px` for "same column" proximity. Tune this based on actual KTP image resolutions in your dataset.

4. **40 samples** is borderline for statistical significance. Results with < 60 samples should be treated as directional, not definitive. Consider McNemar's test if you want to formally test significance on field-level binary outcomes.

5. **LightonOCR hallucination** (the LaTeX math note visible in the raw output) is a real risk for fields it cannot read. Character Precision metric will catch this.

6. **This evaluation measures OCR capability, not end-to-end pipeline performance.** If you later add an LLM parser, re-evaluate separately — do not compare pre-LLM vs post-LLM results on the same metrics.

# KTP Image Validation — CLIP Layer 1 Spec

> Handoff document for the implementing agent. Read the **Failed Approaches** section
> before changing any prompts — most "obvious" fixes have already been tried and they
> do not work. The reasons are explained so you do not repeat them.

---

## 1. Problem

We classify an input image as **`yes`** (a valid KTP photo) or **`no`** (anything else).

The hard part is a specific failure case the client explicitly rejects:

- A **selfie + KTP** (person's face/body visible AND holding a KTP), or
- A **full photo of a person + KTP**

...must be classified **`no`**, even though a KTP *is* physically present in the image.

Only a **KTP on its own** (no person in frame) counts as `yes`. A bare hand holding the
card with **no person/face/body visible** is still `yes`.

The core difficulty: **a KTP is present in BOTH the valid case and the rejected case.**
The thing that actually separates them is "is there a person in the frame" — not "is there
a KTP". CLIP is strong at detecting a KTP and strong at detecting a person, but it is
**weak at negation/composition** (e.g. "a KTP *with no person*"). Any approach that asks
CLIP to reason about the *absence* of a person fails.

---

## 2. Goal — the decision rule (crystal clear)

| Image content | Label |
|---|---|
| KTP alone on a table/surface | `yes` |
| Hand holding KTP, **no person/face/body visible** | `yes` |
| Close-up/scan of a KTP, no person | `yes` |
| **Selfie + KTP** (face/body visible) | `no` |
| **Full person + KTP** (face/body visible) | `no` |
| Random object / room / monitor / blanket / etc. (no KTP) | `no` |
| Selfie / person with no KTP | `no` |

The discriminating axis is **presence of a person**, then **presence of a KTP** — in that
order.

---

## 3. Constraints (non-negotiable)

- **Model:** OpenCLIP **ViT-B/16**, zero-shot. Do not swap models.
- **CPU-only**, fully **offline/local** (data is confidential, no cloud calls).
- **< 10 seconds per image.**
- **CLIP/prompt-only.** Do NOT add a separate face detector (OpenCV/MediaPipe) — this was
  considered and deliberately not used in this layer. See §6 for why it would help and why
  it was declined, but **do not implement it without explicit sign-off.**
- Aggregation is **max** per prompt pool (take the single highest similarity in a pool),
  not mean.

---

## 4. Current approach — TWO-STAGE CLIP decision (this is the working design)

The key structural insight: **do NOT put KTP prompts and person prompts in the same max
pool.** If they compete in one global argmax, the KTP prompt wins on selfie+KTP images
(the KTP caption is "content-rich" and matches strongly), so every selfie+KTP leaks to
`yes`. The fix is to make **two separate CLIP decisions**, each with its own max pool, and
combine them with a rule. KTP prompts must never compete against person prompts in the
same pool.

### Prompt pools

```python
PERSON_PROMPTS = [
    "a selfie of a person",
    "a photo of a person's face and upper body",
    "a person posing for the camera",
]
NO_PERSON_PROMPTS = [
    "a close-up photo of a card or document, no person",
    "an object photographed on its own, no people",
    "a scene with no people in it",
]
KTP_PROMPTS = [
    "a photo of an Indonesian KTP identity card",
    "a close-up of an Indonesian ID card (KTP)",
    "a KTP card on a table or surface",
]
RANDOM_PROMPTS = [
    "a random photo with no ID card",
    "a photo of everyday objects with no card",
]
```

### Decision logic

```python
# Stage 1: IS THERE A PERSON?  (this pool contains NO "KTP" wording on purpose)
person_score    = max(sim(img, PERSON_PROMPTS))
no_person_score = max(sim(img, NO_PERSON_PROMPTS))

if person_score > no_person_score:
    label = "no"          # selfie / person present -> reject. Done.
else:
    # Stage 2: only runs when NO person was detected
    ktp_score    = max(sim(img, KTP_PROMPTS))
    random_score = max(sim(img, RANDOM_PROMPTS))
    label = "yes" if ktp_score > random_score else "no"
```

`sim(img, pool)` = cosine similarity between the image embedding and each text-prompt
embedding in the pool. Encode all text prompts **once at startup** (outside the image
loop) and cache them — text encoding cost does not scale per image, only image encoding
does.

### Why this works (where the previous designs did not)

In Stage 1, `"a selfie of a person"` only competes against `"no person"` prompts. The KTP
prompts are **absent** from this pool, so they cannot steal the argmax. For a selfie+KTP
image, "a person" wins easily over "no person" because a person really is there — the KTP
is irrelevant at this stage and is only checked **after** a person has been ruled out. We
moved the "is there a person" decision into the pool where CLIP is strongest (detecting
people), shielded from the strong KTP signal.

This is a change in **decision structure**, not a change of wording in a shared pool. That
is the difference from every failed attempt below.

---

## 5. Failed approaches — DO NOT RETRY THESE

These were all tried on a ~50-image test set (mix of valid KTP-only, selfie+KTP, and
person+KTP). Each failed for a structural reason, not a wording reason. Re-wording will
not fix them.

### ❌ 5.1 KTP prompts in `yes`, anchored with "alone / by itself / no person"

Example: `"a photo of an Indonesian ID card (KTP) by itself"`,
`"a KTP card ... with no face in the image"`.

**Why it fails:** CLIP weakly weights negation/qualifier phrases ("by itself", "no
person", "alone"). It latches onto the concrete nouns ("KTP", "ID card"). Since a KTP is
genuinely present in selfie+KTP images, these prompts match strongly and the qualifier
barely lowers the score. In a max pool, one such prompt winning = leak to `yes`. These
prompts effectively become **generic "a KTP is present" magnets.**

### ❌ 5.2 Adding composition qualifiers: "flat on a table", "top-down", "scan-like", "lying alone on a surface"

**Why it fails:** Same root cause. CLIP drops locative/compositional qualifiers and hears
"KTP card". A selfie holds the card up flat toward the camera too, so "flat" does not
exclude selfies. Removing one such prompt just moves the false positives to the next
KTP-describing prompt (whack-a-mole). Empirically `"a top-down photo of a KTP card lying
alone on a surface"` produced **10+ new false positives** on pure selfie+KTP images. **Any
prompt that describes a KTP well becomes a magnet for every image containing a KTP,
including selfies.**

### ❌ 5.3 Single shared max pool, with person-detection prompts placed in `no`

Example: `yes` = KTP prompts, `no` = `"a selfie of a person"`, `"a person holding a KTP"`,
etc. — all scored in ONE global argmax.

**Why it fails:** The KTP prompt and the person prompt compete in the same pool. For
selfie+KTP, `"a photo of an Indonesian KTP identity card"` matches **more strongly** than
`"a selfie of a person"` (richer, more specific caption), so the KTP prompt wins the global
argmax and the image is labeled `yes`. Result: **all selfie+KTP images passed as valid.**
No choice of person-prompt wording overcomes this, because the two signals are fighting in
the same pool and KTP wins.

### ❌ 5.4 Anchoring person prompts on "face" (e.g. "a person whose face is visible")

**Why it fails:** The portrait photo printed ON the KTP is itself a human face. Prompts
keyed on "face" get triggered by the KTP's portrait, pushing valid KTP-only images to `no`
(false negatives). Person prompts must key on **action/composition** ("selfie", "posing",
"upper body", "prominently"), never on bare "face".

### Summary of the lesson

> CLIP zero-shot + max cannot separate "KTP alone" vs "KTP + person" **within a single
> pool**, because the KTP is present in both and CLIP cannot be trusted on negation. The
> only fix that works is **structural**: decide "person present?" first, in its own pool,
> with no KTP wording, before checking for a KTP.

---

## 6. Known remaining limitation (and the declined stronger fix)

The two-stage design wins **when the person is reasonably visible in the frame.** The
remaining hole: a **selfie where the KTP fills most of the frame and the face is a small
sliver in a corner.** There, the "person present" signal is weak (few person pixels) and
the image can still pass to Stage 2 and be labeled `yes`. This is far rarer than the
current failures, but it is not fully closable with CLIP-only.

**The robust fix for this last hole (declined for now):** a lightweight face detector
(OpenCV Haar / MediaPipe, CPU, offline) with a **face-size filter** as a pre-gate:
- A detected face **smaller than ~X% of the image area** → ignore it (it is the portrait
  printed on the KTP) → keeps valid KTP-only images safe.
- A detected face that is **large/dominant** → it is a real person → reject as selfie.

This closes both the false-positive (selfie) and false-negative (KTP portrait) cases
deterministically, because the printed portrait is always small and a real selfie face is
always large. It was declined to keep this layer CLIP/prompt-only. **Do not add it without
explicit sign-off**, but it is documented here as the correct next step if CLIP-only
false-positives remain unacceptable.

---

## 7. Testing guidance

- Test on the labeled set (mix of KTP-only, selfie+KTP, person+KTP, random objects).
- **Log which prompt won** (the argmax prompt index) for every misclassified image. A
  single prompt that keeps winning on wrong images is a bad prompt — investigate it
  specifically rather than rewriting everything.
- Report **FP/FN broken down per prompt**, not just an aggregate accuracy number. That
  breakdown is what tells you whether the remaining errors are the known "KTP-dominant,
  face-tiny" edge case (§6) or a genuine prompt problem.
- If valid KTP-only images start getting detected as "person" in Stage 1, narrow
  `PERSON_PROMPTS` toward "selfie" only and drop the broader "face/upper body" prompt.
- Keep prompt-pool sizes balanced within each stage's two pools — in max aggregation, a
  pool with more prompts gets a statistical edge (more chances at a high similarity).

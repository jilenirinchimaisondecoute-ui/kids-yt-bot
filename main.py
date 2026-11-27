"""Pipeline: story + song (Gemini) -> images (Cloudflare) -> rhythmic video (FFmpeg)
-> beat music + Arabic chant voice -> YouTube upload.

Env vars: GEMINI_API_KEY, CF_ACCOUNT_ID, CF_API_TOKEN, YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN
Options:  DRY_RUN=1 (no upload), PRIVACY=private|unlisted|public
"""
import asyncio, base64, glob, hashlib, io, json, math, os, random, subprocess, sys, time, wave
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path("output"); OUT.mkdir(exist_ok=True)

BPM = 120
BEAT = 60 / BPM                  # 0.5 s
SCENES = 15
UNIQUE_IMAGES = 8                # one picture serves ~2 scenes (saves the free daily quota)
SCENE_SECONDS = BEAT * 8         # 4 s = 2 bars -> 15 x 4 = 60 s
TOTAL = SCENES * SCENE_SECONDS
FPS = 25
SR = 44100

TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-3.6-flash")
PRIVACY = os.getenv("PRIVACY", "private")
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
VOICES = ["ar-EG-SalmaNeural", "ar-SA-ZariyahNeural"]

STYLE = ("Pixar-style 3D cartoon render, vibrant saturated colors, soft neutral daylight, "
         "no orange or golden color cast on the fur, cute and child-friendly, high detail, "
         "wide cinematic landscape shot, the puppy is in the center of the frame")
CHARACTER = ("Bobo, a small fluffy puppy with soft white-cream fur, the same fur color in every picture, "
             "and a white chest patch, huge round eyes with warm amber-brown irises and black pupils, "
             "both eyes exactly the same color, floppy light-brown ears, thin dark-brown eyebrows, "
             "a small red scarf around his neck and a short wagging tail")
THEMES = ["finding a glowing mushroom", "helping a lost baby firefly", "crossing a rainbow bridge",
          "meeting a friendly owl", "a treasure of shiny acorns", "dancing with forest butterflies",
          "rescuing a kite from a tree", "a picnic with bunny friends", "a splashy puddle jumping party",
          "following a trail of magic stars"]


def log(*a): print("[bot]", *a, flush=True)
def pause(s): time.sleep(s)
def clean(value): return "".join(str(value).split())


def retry(fn, *a, tries=6, **k):
    for n in range(tries):
        try:
            return fn(*a, **k)
        except Exception as e:
            log(f"{fn.__name__} failed ({n + 1}/{tries}): {e}")
            if n == tries - 1:
                raise
            pause(20 * (n + 1))


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])


# ---------------------------------------------------------------- story + song
def text_models(client):
    """Preferred model first, then any other Gemini flash text model the key can use."""
    names = [TEXT_MODEL]
    try:
        for m in client.models.list():
            n = m.name.replace("models/", "")
            acts = getattr(m, "supported_actions", None) or ["generateContent"]
            bad = ("image", "tts", "live", "audio", "embedding", "native", "robotics", "computer")
            if "generateContent" in acts and "flash" in n and not any(x in n for x in bad) and n not in names:
                names.append(n)
    except Exception as e:
        log("model list failed:", e)
    return names[:5]


def make_story():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=clean(os.environ["GEMINI_API_KEY"]))
    theme = random.choice(THEMES)
    prompt = f"""Write a short, joyful kids' song-story for toddlers (ages 2-6), set in a magic forest.
Hero: {CHARACTER}. Theme today: {theme}. Seed: {date.today().isoformat()}-{random.randint(0, 9999)}.
Return JSON: {{"title_ar": str (<=70 chars, Arabic, catchy, child-safe),
"title_en": str, "description_ar": str (2 short sentences about the story), "tags": [str x8],
"scenes": [{SCENES} objects {{"image_prompt": str, "lyric_ar": str}}]}}.
Rules for scenes: they tell ONE simple story in order (start, small problem, friends help, happy ending).
Each image_prompt: ONE clear scene in English, hero always visible and doing an ACTIVE pose
(jumping, running, dancing, waving, rolling, sniffing), lively and colorful, no text in the image,
no scary elements.
Rules for lyric_ar: the Arabic song line for that scene, simple modern Arabic that toddlers understand,
2 to 6 words, fun rhythm, ends with rhyme (couplets). The lyric must match what happens in that scene
and the lines together must tell the story. Scenes 4, 8 and 12 use the SAME catchy chorus line that
repeats the hero name Bobo (بوبو). Do not use diacritics. No English in lyrics."""
    models = text_models(client)
    log("text models:", models)
    last = None
    for rnd in range(3):
        for model in models:
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"))
                story = json.loads(r.text)
                assert len(story["scenes"]) >= SCENES
                story["scenes"] = story["scenes"][:SCENES]
                for sc in story["scenes"]:
                    sc["lyric_ar"] = str(sc.get("lyric_ar", "")).strip()
                log("story model:", model)
                return story
            except Exception as e:
                last = e
                log(f"{model} failed (round {rnd + 1}): {str(e)[:160]}")
                pause(8)
        pause(30)
    raise last


# ---------------------------------------------------------------- images
def _cf_image(prompt):
    import requests
    acc = clean(os.environ["CF_ACCOUNT_ID"])
    tok = clean(os.environ["CF_API_TOKEN"])
    url = f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    r = requests.post(url, headers={"Authorization": f"Bearer {tok}"},
                      json={"prompt": prompt[:2000], "steps": 4}, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Cloudflare {r.status_code}: {r.text[:600]}")
    b64 = r.json()["result"]["image"]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def to_16x9(img, size=(2880, 1620)):
    w, h = img.size
    target = 16 / 9
    if w / h > target:
        nw = int(h * target); img = img.crop(((w - nw) // 2, 0, (w + nw) // 2, h))
    else:
        nh = int(w / target); img = img.crop((0, (h - nh) // 2, w, (h + nh) // 2))
    return img.resize(size, Image.LANCZOS)


def image_map():
    """scene index -> unique image index (15 scenes share 8 pictures)."""
    return [i * UNIQUE_IMAGES // SCENES for i in range(SCENES)]


def make_images(story):
    mp = image_map()
    uniq = []                                   # scene index whose prompt draws each unique image
    for i in range(SCENES):
        if i == 0 or mp[i] != mp[i - 1]:
            uniq.append(i)
    files = []
    for k, si in enumerate(uniq):
        full = f"{CHARACTER}. {story['scenes'][si]['image_prompt']}. {STYLE}"
        img = None
        for attempt in range(3):
            try:
                img = _cf_image(full)
                break
            except Exception as e:
                log(f"image {k} attempt {attempt} failed: {e}")
                if "429" in str(e):             # daily free quota finished: do not wait
                    break
                pause(15)
        if img is None:
            if not files:
                raise RuntimeError(f"image {k} failed (Cloudflare daily quota may be used up)")
            log(f"image {k}: reusing previous image")
            files.append(files[-1]); continue
        p = OUT / f"img_{k}.png"; to_16x9(img).save(p); files.append(p)
        log("image", k, "ok")
        pause(1)
    return [files[mp[i]] for i in range(SCENES)]


# ---------------------------------------------------------------- lyrics on screen
def ar(text):
    try:
        from PIL import features
        if features.check("raqm"):      # Pillow already shapes Arabic correctly
            return text
    except Exception:
        pass
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def find_font():
    pats = ["/usr/share/fonts/**/NotoSansArabic*Bold*.ttf", "/usr/share/fonts/**/NotoSansArabic*.ttf",
            "/usr/share/fonts/**/Amiri*Bold*.ttf", "/usr/share/fonts/**/Amiri*.ttf",
            "/usr/share/fonts/**/DejaVuSans-Bold.ttf", "/usr/share/fonts/**/DejaVuSans.ttf"]
    for pat in pats:
        found = sorted(glob.glob(pat, recursive=True))
        if found:
            return found[0]
    return None


def add_lyric(src, text, dst):
    """Draw the lyric line inside the safe area (the video zooms in a little)."""
    img = Image.open(src).convert("RGBA")
    W, H = img.size
    if not text:
        img.convert("RGB").save(dst); return
    font_path = find_font()
    draw = ImageDraw.Draw(img)

    def font(size):
        return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()

    max_w = int(W * 0.70)
    words = text.split()
    layouts = [[text]]
    if len(words) > 2:
        mid = (len(words) + 1) // 2
        layouts.append([" ".join(words[:mid]), " ".join(words[mid:])])
    chosen = None
    for lines in layouts:
        size = 150
        while size >= 80:
            f = font(size)
            widths = [draw.textlength(ar(t), font=f) for t in lines]
            if max(widths) <= max_w:
                chosen = (lines, f, size); break
            size -= 6
        if chosen:
            break
    if not chosen:
        lines = layouts[-1]; f = font(80); chosen = (lines, f, 80)
    lines, f, size = chosen
    line_h = int(size * 1.35)
    block_h = line_h * len(lines)
    cy = int(H * 0.80)
    top = cy - block_h // 2
    pad = 40
    block_w = int(max(draw.textlength(ar(t), font=f) for t in lines))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle((W // 2 - block_w // 2 - pad * 1.5, top - pad // 2,
                          W // 2 + block_w // 2 + pad * 1.5, top + block_h + pad // 2),
                         radius=60, fill=(30, 20, 80, 150))
    img = Image.alpha_composite(img, layer)
    draw = ImageDraw.Draw(img)
    for k, t in enumerate(lines):
        draw.text((W // 2, top + line_h * k + line_h // 2), ar(t), font=f, anchor="mm",
                  fill=(255, 235, 59), stroke_width=7, stroke_fill=(60, 20, 90))
    img.convert("RGB").save(dst)


def add_lyrics_all(story, images):
    out = []
    for i, (img, sc) in enumerate(zip(images, story["scenes"])):
        dst = OUT / f"scene_{i}_text.png"
        try:
            add_lyric(img, sc.get("lyric_ar", ""), dst)
        except Exception as e:
            log(f"lyric overlay {i} failed: {e}")
            dst = img
        out.append(dst)
    return out


# ---------------------------------------------------------------- music (120 BPM kids beat)
def _kick():
    n = int(0.28 * SR); t = np.arange(n) / SR
    ph = 2 * math.pi * np.cumsum(50 + 90 * np.exp(-t * 28)) / SR
    return np.sin(ph) * np.exp(-t * 9)


def _noise(rng, dur, decay, hp=True):
    n = int(dur * SR); t = np.arange(n) / SR
    x = rng.standard_normal(n)
    if hp:
        x = np.diff(x, prepend=0)
    return x * np.exp(-t * decay)


def _pluck(f, dur):
    n = int(dur * SR); t = np.arange(n) / SR
    s = np.sin(2 * math.pi * f * t) + 0.35 * np.sin(4 * math.pi * f * t) + 0.12 * np.sin(6 * math.pi * f * t)
    return s * np.exp(-7 * t) * np.minimum(1, t * 400)


def _bass(f, dur):
    n = int(dur * SR); t = np.arange(n) / SR
    s = np.sin(2 * math.pi * f * t) + 0.45 * np.sin(4 * math.pi * f * t)
    return s * np.exp(-3.2 * t) * np.minimum(1, t * 300)


def _bell(f, dur=1.4):
    n = int(dur * SR); t = np.arange(n) / SR
    s = np.sin(2 * math.pi * f * t) + 0.5 * np.sin(2 * math.pi * f * 2.76 * t)
    return s * np.exp(-3 * t)


def _add(buf, start_s, sound, gain):
    a = int(start_s * SR)
    if a >= len(buf):
        return
    b = min(len(buf), a + len(sound))
    buf[a:b] += sound[:b - a] * gain


def make_music(seed_text="bobo"):
    rng = np.random.default_rng(int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16))
    buf = np.zeros(int((TOTAL + 3) * SR))
    kick, clap, hat = _kick(), None, None
    roots = [130.81, 110.00, 87.31, 98.00]                    # C  Am  F  G
    chords = [[523.25, 659.25, 783.99, 1046.50], [440.00, 523.25, 659.25, 880.00],
              [349.23, 440.00, 523.25, 698.46], [392.00, 493.88, 587.33, 783.99]]
    motif = [int(rng.integers(0, 4)) for _ in range(8)]
    rests = {3, 7} if rng.random() < 0.5 else {2, 7}
    bars = int(TOTAL / (BEAT * 4)) + 2
    for bar in range(bars):
        b0 = bar * BEAT * 4
        ci = bar % 4
        for beat in range(4):
            t0 = b0 + beat * BEAT
            _add(buf, t0, kick, 0.95)
            if beat in (1, 3):
                _add(buf, t0, _noise(rng, 0.14, 38), 0.30)
            _add(buf, t0, _noise(rng, 0.05, 90), 0.07)
            _add(buf, t0 + BEAT / 2, _noise(rng, 0.05, 90), 0.14)
        for t0, f in ((b0, roots[ci]), (b0 + BEAT * 1.5, roots[ci]), (b0 + BEAT * 2.5, roots[ci] * 1.5),
                      (b0 + BEAT * 3, roots[ci])):
            _add(buf, t0, _bass(f, 0.45), 0.38)
        seq = motif if bar % 4 != 3 else motif[::-1]
        for j, idx in enumerate(seq):
            if j in rests:
                continue
            _add(buf, b0 + j * BEAT / 2, _pluck(chords[ci][idx], 0.4), 0.13)
        if bar % 2 == 0:
            _add(buf, b0, _bell(chords[ci][3] * 2), 0.07)
    buf = buf[:int(TOTAL * SR)]
    buf = buf / max(1e-9, np.abs(buf).max()) * 0.9
    return buf.astype(np.float32)


# ---------------------------------------------------------------- voice (Arabic chant)
def _edge_tts(text, path, voice, rate, pitch):
    import edge_tts

    async def go():
        await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(path))
    asyncio.run(go())


def tts_line(text, path, rate="+8%", pitch="+25Hz"):
    for voice in VOICES:
        for _ in range(2):
            try:
                _edge_tts(text, path, voice, rate, pitch)
                if Path(path).exists() and Path(path).stat().st_size > 1000:
                    return True
            except Exception as e:
                log(f"edge-tts {voice} failed: {e}")
                pause(3)
    try:
        from gtts import gTTS
        gTTS(text, lang="ar").save(str(path))
        return Path(path).exists() and Path(path).stat().st_size > 1000
    except Exception as e:
        log("gTTS failed:", e)
        return False


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return data


def clean_wav(src):
    """mp3 -> mono wav with leading/trailing silence removed."""
    wav = Path(str(src) + ".wav")
    run(["ffmpeg", "-y", "-i", str(src), "-ar", str(SR), "-ac", "1", "-af",
         "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
         "silenceremove=start_periods=1:start_threshold=-45dB,areverse", str(wav)])
    return wav


def estimate_f0(x):
    """Median pitch (Hz) of the voiced part of a short mono signal."""
    frame = int(0.08 * SR); hop = frame // 2
    lo, hi = int(SR / 420), int(SR / 80)
    f0s = []
    for a in range(0, max(1, len(x) - frame), hop):
        seg = x[a:a + frame]
        if len(seg) < frame or np.sqrt(np.mean(seg ** 2)) < 0.02:
            continue
        seg = seg - seg.mean()
        spec = np.fft.rfft(seg, 2 * frame)
        ac = np.fft.irfft(spec * np.conj(spec))[:frame]
        if ac[0] <= 0:
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] > 0.35:
            f0s.append(SR / lag)
    return float(np.median(f0s)) if f0s else None


def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def find_syllables(x):
    """Split a speech signal into syllable chunks at energy valleys. Returns [(start, end)] in samples."""
    hop = int(0.01 * SR)
    n = len(x) // hop
    if n < 8:
        return [(0, len(x))]
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1))
    env = np.convolve(rms, np.ones(5) / 5, "same")
    thr = 0.12 * env.max()
    peaks = [i for i in range(1, n - 1) if env[i] >= env[i - 1] and env[i] > env[i + 1] and env[i] > thr]
    keep = []
    for p in peaks:
        if keep and p - keep[-1] < 11:
            if env[p] > env[keep[-1]]:
                keep[-1] = p
        else:
            keep.append(p)
    merged = True
    while merged and len(keep) > 1:
        merged = False
        for j in range(len(keep) - 1):
            v = keep[j] + int(np.argmin(env[keep[j]:keep[j + 1] + 1]))
            if env[v] > 0.7 * min(env[keep[j]], env[keep[j + 1]]):
                if env[keep[j]] >= env[keep[j + 1]]:
                    del keep[j + 1]
                else:
                    del keep[j]
                merged = True
                break
    if not keep:
        return [(0, len(x))]
    cuts = [0]
    for j in range(len(keep) - 1):
        cuts.append(keep[j] + int(np.argmin(env[keep[j]:keep[j + 1] + 1])))
    cuts.append(n)
    return [(cuts[j] * hop, min(len(x), cuts[j + 1] * hop)) for j in range(len(cuts) - 1)]


def shift_stretch(x, ratio, stretch, vibrato=False):
    """Change pitch by `ratio` (keeping the voice timbre) and duration by `stretch`."""
    src = OUT / "chunk_in.wav"; dst = OUT / "chunk_out.wav"
    write_wav(src, x)
    tail = ",vibrato=f=5.5:d=0.07" if vibrato else ""
    try:
        run(["ffmpeg", "-y", "-i", str(src), "-af",
             f"rubberband=pitch={ratio:.4f}:tempo={1 / stretch:.4f}:formant=preserved:transients=smooth{tail}",
             str(dst)])
    except Exception:
        tempo = 1 / (ratio * stretch)
        chain = []
        while tempo > 2.0:
            chain.append("atempo=2.0"); tempo /= 2.0
        while tempo < 0.5:
            chain.append("atempo=0.5"); tempo /= 0.5
        chain.append(f"atempo={tempo:.4f}")
        run(["ffmpeg", "-y", "-i", str(src), "-af",
             f"asetrate={SR * ratio:.1f},aresample={SR}," + ",".join(chain), str(dst)])
    y = read_wav(dst)
    k = min(len(y) // 4, int(0.01 * SR))
    if k > 0:
        y[:k] *= np.linspace(0, 1, k); y[-k:] *= np.linspace(1, 0, k)
    return y


# pentatonic melody (G A C D E G) in the natural range of the voice, and a few singable patterns
SCALE = [196.00, 220.00, 261.63, 293.66, 329.63, 392.00]
PATTERNS = [[0, 1, 2, 3, 2, 1, 2, 0], [2, 3, 4, 3, 2, 1, 0, 1], [0, 2, 3, 2, 4, 3, 2, 0], [3, 2, 1, 2, 3, 4, 3, 2]]
CHORD_ROOTS = [261.63, 220.00, 174.61, 196.00]          # C Am F G, like the music


def note_for(k, n, pattern, chord_root):
    if k == n - 1:                                       # last syllable rests on the chord root
        f = chord_root
        while f < 196: f *= 2
        while f > 392: f /= 2
        return f
    return SCALE[pattern[k % len(pattern)]]


def positions(n):
    """Start of each syllable inside a 2-bar scene, in eighth notes (0..15)."""
    return [int(k * 15 / n) if n > 1 else 0 for k in range(n)]


def sing_line(wav, scene_idx, text):
    """Pitch every syllable of one spoken line onto a melody note and hold it on the beat."""
    x = read_wav(wav)
    syl = find_syllables(x)[:14]
    n = len(syl)
    pat = PATTERNS[int(hashlib.md5(text.encode()).hexdigest()[:4], 16) % len(PATTERNS)]
    root = CHORD_ROOTS[(2 * scene_idx + 1) % 4]
    f_line = estimate_f0(x) or 210.0
    pos = positions(n)
    buf = np.zeros(int(SCENE_SECONDS * SR) + SR)
    notes = []
    for k, (a, b) in enumerate(syl):
        seg = x[a:b]
        if len(seg) < int(0.05 * SR):
            continue
        f0 = estimate_f0(seg) or f_line
        target = note_for(k, n, pat, root)
        ratio = float(np.clip(target / f0, 0.6, 2.0))
        nxt = pos[k + 1] if k + 1 < n else 16
        hold = max(0.25, (nxt - pos[k]) * BEAT / 2 - 0.03)
        nat = len(seg) / SR
        stretch = float(np.clip(hold / nat, 0.8, 3.0))
        y = shift_stretch(seg, ratio, stretch, vibrato=stretch > 1.8)
        start = int(max(0, pos[k] * BEAT / 2 - 0.05) * SR)
        buf[start:start + len(y)] += y[:len(buf) - start]
        notes.append((pos[k] * BEAT / 2, min(hold, len(y) / SR), target))
    peak = np.abs(buf).max()
    if peak <= 0 or not notes:
        raise RuntimeError("empty line")
    return buf / peak * 0.9, notes


def fit_line(wav, lo=2.3, hi=3.5):
    """Fallback: plain spoken line stretched into the scene."""
    data = read_wav(wav)
    dur = len(data) / SR
    target = min(max(dur, lo), hi)
    tempo = float(np.clip(dur / target, 0.65, 1.7))
    out = Path(str(wav) + ".fit.wav")
    run(["ffmpeg", "-y", "-i", str(wav), "-af", f"atempo={tempo:.4f},aecho=0.8:0.6:45:0.25", str(out)])
    y = read_wav(out)
    peak = np.abs(y).max()
    return y / peak * 0.9 if peak > 0 else y


def words_only_notes(text, scene_idx):
    """No voice available: still give the scene a melody (about 2 notes per word)."""
    n = max(2, len(text.split()) * 2)
    pat = PATTERNS[int(hashlib.md5(text.encode()).hexdigest()[:4], 16) % len(PATTERNS)]
    root = CHORD_ROOTS[(2 * scene_idx + 1) % 4]
    pos = positions(n)
    notes = []
    for k in range(n):
        nxt = pos[k + 1] if k + 1 < n else 16
        notes.append((pos[k] * BEAT / 2, max(0.25, (nxt - pos[k]) * BEAT / 2 - 0.03), note_for(k, n, pat, root)))
    return notes


def make_voices(story):
    """Returns (voices, leads): sung lines placed on their scene + melody notes for a lead instrument."""
    voices, leads, cache = [], [], {}
    for i, sc in enumerate(story["scenes"]):
        text = sc.get("lyric_ar", "")
        if not text:
            continue
        if text not in cache:
            mp3 = OUT / f"line_{len(cache)}.mp3"
            audio, notes = None, None
            try:
                if tts_line(text, mp3, rate="-10%", pitch="+0Hz"):
                    wav = clean_wav(mp3)
                    try:
                        audio, notes = sing_line(wav, i, text)
                    except Exception as e:
                        log(f"sing line {i} failed ({e}); using spoken fallback")
                        audio = fit_line(wav)
                        notes = words_only_notes(text, i)
            except Exception as e:
                log(f"voice {i} failed: {e}")
            if notes is None:
                notes = words_only_notes(text, i)
            cache[text] = (audio, notes)
        audio, notes = cache[text]
        if audio is not None:
            voices.append((i * SCENE_SECONDS, audio))
        leads.extend((i * SCENE_SECONDS + t, d, f) for t, d, f in notes)
    log("sung lines:", len(voices), "/", SCENES)
    return voices, leads


def _lead(f, dur):
    n = int(dur * SR); t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * math.pi * 5.5 * t) * np.minimum(1, t * 4)
    ph = 2 * math.pi * np.cumsum(f * vib) / SR
    s = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
    return s * np.minimum(1, t * 60) * np.clip((dur - t) * 25, 0, 1)


def add_leads(music, leads, gain):
    buf = music.astype(np.float64).copy()
    for t, d, f in leads:
        _add(buf, t, _lead(f, d), gain)
    return buf.astype(np.float32)


def mix_audio(music, voices):
    total = int(TOTAL * SR)
    track = np.zeros(total, dtype=np.float32)
    env = np.ones(total, dtype=np.float32)
    for start, v in voices:
        a = int(start * SR); b = min(total, a + len(v))
        track[a:b] += v[:b - a]
        env[max(0, a - int(0.1 * SR)):min(total, b + int(0.15 * SR))] = 0.5
    k = int(0.12 * SR)
    padded = np.pad(env, (k // 2, k - k // 2 - 1), mode="edge")
    cs = np.cumsum(np.insert(padded, 0, 0))
    smooth = ((cs[k:] - cs[:-k]) / k).astype(np.float32)
    mixed = music[:total] * smooth * 0.75 + track
    mixed = mixed / max(1.0, np.abs(mixed).max() / 0.95)
    p = OUT / "mix.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mixed * 32767).astype(np.int16).tobytes())
    return p


# ---------------------------------------------------------------- video
def make_video(images, mix, same_as_prev=None):
    frames = int(SCENE_SECONDS * FPS)
    beat_f = BEAT * FPS
    clips = []
    for i, img in enumerate(images):
        drift = f"0.05*on/{frames}" if i % 2 == 0 else f"0.05*(1-on/{frames})"
        base = 1.22 if (same_as_prev and same_as_prev[i]) else 1.10
        z = f"{base}+0.05*pow(cos(PI*on/{beat_f}),2)+{drift}"
        s = 1 if i % 2 == 0 else -1
        x = f"(iw-iw/zoom)/2+iw*0.025*sin(2*PI*on/{frames})*{s}"
        y = f"(ih-ih/zoom)/2+ih*0.015*cos(2*PI*on/{frames})"
        vf = (f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s=1920x1080:fps={FPS},"
              f"fade=t=in:st=0:d=0.16:color=white,eq=saturation=1.1,format=yuv420p")
        c = OUT / f"clip_{i}.mp4"
        run(["ffmpeg", "-y", "-loop", "1", "-i", str(img), "-vf", vf, "-t", str(SCENE_SECONDS),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", str(FPS), str(c)])
        clips.append(c)
    lst = OUT / "list.txt"
    lst.write_text("".join(f"file '{c.name}'\n" for c in clips))
    silent = OUT / "silent.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent)])
    final = OUT / "final.mp4"
    run(["ffmpeg", "-y", "-i", str(silent), "-i", str(mix), "-t", str(TOTAL),
         "-af", f"afade=t=in:d=0.5,afade=t=out:st={TOTAL - 1.5}:d=1.5,loudnorm=I=-16",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ac", "2", str(final)])
    return final


# ---------------------------------------------------------------- youtube
def upload(video, story, thumb):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    creds = Credentials(None, refresh_token=clean(os.environ["YT_REFRESH_TOKEN"]),
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=clean(os.environ["YT_CLIENT_ID"]),
                        client_secret=clean(os.environ["YT_CLIENT_SECRET"]),
                        scopes=["https://www.googleapis.com/auth/youtube.upload"])
    yt = build("youtube", "v3", credentials=creds)
    lines, seen = [], set()
    for sc in story["scenes"]:
        t = sc.get("lyric_ar", "")
        if t and t not in seen:
            seen.add(t); lines.append(t)
    desc = (story["description_ar"] + "\n\n🎵 كلمات الأنشودة:\n" + "\n".join(lines) +
            "\n\n" + story["title_en"] + "\n\nAI-generated animation for kids.")
    body = {"snippet": {"title": story["title_ar"][:95], "description": desc[:4500], "tags": story["tags"],
                        "categoryId": "1", "defaultLanguage": "ar"},
            "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": True,
                       "containsSyntheticMedia": True}}
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(video), chunksize=-1, resumable=True))
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    vid = resp["id"]; log("uploaded https://youtu.be/" + vid)
    try:
        yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb))).execute()
    except Exception as e:
        log("thumbnail skipped:", e)
    return vid


def custom_mix():
    """If song.mp3 / song.wav / song.m4a exists in the repo, use it as the whole soundtrack."""
    for name in ("song.mp3", "song.wav", "song.m4a"):
        if Path(name).exists():
            p = OUT / "mix.wav"
            run(["ffmpeg", "-y", "-i", name, "-t", str(TOTAL), "-ar", str(SR), "-ac", "1", str(p)])
            log("using custom song:", name)
            return p
    return None


def melody_mix(story):
    """Instrument melody for every lyric line, on top of the beat. No synthetic voice."""
    leads = []
    for i, sc in enumerate(story["scenes"]):
        text = sc.get("lyric_ar", "")
        if text:
            leads.extend((i * SCENE_SECONDS + t, d, f) for t, d, f in words_only_notes(text, i))
    music = add_leads(make_music(story["title_en"]), leads, 0.5)
    return mix_audio(music, [])


def main():
    story = make_story(); log("story:", story["title_en"])
    (OUT / "story.json").write_text(json.dumps(story, ensure_ascii=False, indent=2))
    imgs = make_images(story)
    texted = add_lyrics_all(story, imgs)
    mix = custom_mix() or melody_mix(story)
    mp = image_map()
    video = make_video(texted, mix, [i > 0 and mp[i] == mp[i - 1] for i in range(SCENES)]); log("video ready", video)
    thumb = OUT / "thumb.jpg"; Image.open(imgs[2]).resize((1280, 720)).save(thumb, quality=90)
    if DRY_RUN:
        log("DRY_RUN: upload skipped"); return
    upload(video, story, thumb)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FAILED:", e); sys.exit(1)

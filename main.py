"""Pipeline: story (Gemini) -> images (Cloudflare FLUX) -> animated video (FFmpeg) -> music -> YouTube upload.

Env vars: GEMINI_API_KEY, CF_ACCOUNT_ID, CF_API_TOKEN, HF_TOKEN (optional),
          YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN
Options:  DRY_RUN=1 (no upload), PRIVACY=private|unlisted|public, IMAGE_PROVIDER=cf|gemini|hf
"""
import base64, io, json, math, os, random, subprocess, sys, wave
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("output"); OUT.mkdir(exist_ok=True)
SCENES = 8
SCENE_SECONDS = 7.5          # 8 x 7.5 = 60 s
FPS = 25
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-3.6-flash")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-3.1-flash-image")
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "cf")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
PRIVACY = os.getenv("PRIVACY", "private")
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

STYLE = ("Pixar-style 3D cartoon render, vibrant saturated colors, soft cinematic lighting, "
         "cute and child-friendly, high detail, wide cinematic landscape shot, "
         "the puppy is in the center of the frame")
CHARACTER = ("Bobo, a small fluffy cream-colored puppy with a white chest patch, huge round brown eyes, "
             "floppy light-brown ears, a small red scarf around his neck and a short wagging tail")
THEMES = ["finding a glowing mushroom", "helping a lost baby firefly", "crossing a rainbow bridge",
          "meeting a friendly owl", "a treasure of shiny acorns", "dancing with forest butterflies",
          "rescuing a kite from a tree", "a picnic with bunny friends"]


def log(*a): print("[bot]", *a, flush=True)


def pause(s): import time; time.sleep(s)


def retry(fn, tries=6):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                raise
            wait = 20 * (k + 1)
            log(f"retry {k+1}/{tries} after error: {str(e)[:100]} - waiting {wait}s")
            pause(wait)


# ---------------------------------------------------------------- story
def make_story():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    theme = random.choice(THEMES)
    prompt = f"""Write a gentle wordless adventure for toddlers (ages 2-6) in a magic forest.
Hero: {CHARACTER}. Theme today: {theme}. Seed: {date.today().isoformat()}-{random.randint(0, 9999)}.
Return JSON: {{"title_ar": str (<=70 chars, Arabic, catchy, child-safe),
"title_en": str, "description_ar": str (2 short sentences), "tags": [str x8],
"scenes": [{SCENES} objects {{"image_prompt": str, "motion": "in"|"out"}}]}}.
Each image_prompt describes ONE clear scene in English (max 40 words), always featuring the puppy,
no text in image, happy mood, no scary elements. Scenes must follow a simple story: start, small problem, friends, joy."""
    r = client.models.generate_content(
        model=TEXT_MODEL, contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"))
    story = json.loads(r.text)
    assert len(story["scenes"]) >= SCENES
    story["scenes"] = story["scenes"][:SCENES]
    return story


# ---------------------------------------------------------------- images
def _cf_image(prompt, seed):
    import requests
    acc = os.environ["CF_ACCOUNT_ID"]
    tok = os.environ["CF_API_TOKEN"]
    url = f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    body = {"prompt": prompt[:2000], "steps": 6, "seed": seed}
    r = requests.post(url, headers={"Authorization": f"Bearer {tok}"}, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"cloudflare {r.status_code}: {r.text[:300]}")
    data = r.json()
    b64 = data["result"]["image"]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _gemini_image(prompt, ref_bytes=None):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    contents = []
    if ref_bytes:
        contents.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/png"))
        contents.append("Keep the exact same character design as the reference image. " + prompt)
    else:
        contents.append(prompt)
    r = client.models.generate_content(
        model=IMAGE_MODEL, contents=contents,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]))
    for part in r.candidates[0].content.parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    raise RuntimeError("no image returned")


def _hf_image(prompt):
    from huggingface_hub import InferenceClient
    c = InferenceClient(token=os.getenv("HF_TOKEN"))
    return c.text_to_image(prompt, model=HF_IMAGE_MODEL, width=1344, height=768).convert("RGB")


def to_16x9(img, size=(2880, 1620)):
    w, h = img.size
    target = 16 / 9
    if w / h > target:
        nw = int(h * target); img = img.crop(((w - nw) // 2, 0, (w + nw) // 2, h))
    else:
        nh = int(w / target); img = img.crop((0, (h - nh) // 2, w, (h + nh) // 2))
    return img.resize(size, Image.LANCZOS)


def make_images(story):
    paths, ref = [], None
    seed = random.randint(1000, 999999)
    for i, sc in enumerate(story["scenes"]):
        full = f"{CHARACTER}. {sc['image_prompt']}. {STYLE}"
        img = None
        for attempt in range(3):
            try:
                if IMAGE_PROVIDER == "cf":
                    img = _cf_image(full, seed)
                elif IMAGE_PROVIDER == "gemini":
                    img = _gemini_image(full, ref)
                else:
                    img = _hf_image(full)
                break
            except Exception as e:
                log(f"image {i} attempt {attempt} failed: {e}")
                pause(15)
        if img is None:
            raise RuntimeError(f"scene {i} image failed")
        if i == 0 and IMAGE_PROVIDER == "gemini":
            b = io.BytesIO(); img.save(b, "PNG"); ref = b.getvalue()
        p = OUT / f"scene_{i}.png"; to_16x9(img).save(p); paths.append(p)
        log("image", i, "ok")
    return paths


# ---------------------------------------------------------------- music
def hf_music(prompt, seconds=30):
    try:
        from huggingface_hub import InferenceClient
        data = InferenceClient(token=os.getenv("HF_TOKEN")).text_to_audio(
            prompt, model="facebook/musicgen-small")
        p = OUT / "music_hf.flac"
        p.write_bytes(data.audio if hasattr(data, "audio") else data)
        return p
    except Exception as e:
        log("musicgen unavailable:", e)
        return None


def procedural_music(seconds=64, bpm=118, sr=44100):
    """Cheerful C-major pentatonic loop with bass, bouncy rhythm. 100% offline fallback."""
    scale = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25]
    beat = 60 / bpm / 2
    n = int(seconds / beat)
    rnd = random.Random(date.today().toordinal())
    out = np.zeros(int(seconds * sr) + sr)
    melody = [rnd.choice(scale) for _ in range(8)]
    for i in range(n):
        t0 = int(i * beat * sr)
        f = melody[i % 8] if i % 16 < 8 else melody[(i * 3) % 8]
        L = int(beat * sr * 0.9)
        t = np.arange(L) / sr
        env = np.exp(-5 * t / (beat * 0.9))
        note = (np.sin(2 * math.pi * f * t) + 0.3 * np.sin(4 * math.pi * f * t)) * env * 0.35
        out[t0:t0 + L] += note
        if i % 4 == 0:  # bass
            b = 130.81 if (i // 4) % 2 == 0 else 98.0
            bl = int(beat * 4 * sr * 0.9)
            tb = np.arange(bl) / sr
            out[t0:t0 + bl] += np.sin(2 * math.pi * b * tb) * np.exp(-2 * tb) * 0.35
        if i % 2 == 1:  # soft tick
            tk = np.random.default_rng(i).standard_normal(int(0.03 * sr)) * np.exp(-np.linspace(0, 8, int(0.03 * sr))) * 0.08
            out[t0:t0 + len(tk)] += tk
    out = out[: int(seconds * sr)]
    out = (out / max(1e-9, np.abs(out).max()) * 0.8 * 32767).astype(np.int16)
    p = OUT / "music_proc.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(out.tobytes())
    return p


# ---------------------------------------------------------------- video
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])


def make_video(story, images, music):
    frames = int(SCENE_SECONDS * FPS)
    clips = []
    for i, (img, sc) in enumerate(zip(images, story["scenes"])):
        if sc.get("motion", "in") == "in":
            z = f"min(1+0.25*on/{frames},1.25)"
        else:
            z = f"max(1.25-0.25*on/{frames},1.0)"
        vf = (f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={FPS},"
              f"fade=t=in:st=0:d=0.4,fade=t=out:st={SCENE_SECONDS-0.4}:d=0.4,format=yuv420p")
        c = OUT / f"clip_{i}.mp4"
        run(["ffmpeg", "-y", "-loop", "1", "-i", str(img), "-vf", vf, "-t", str(SCENE_SECONDS),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(c)])
        clips.append(c)
    lst = OUT / "list.txt"
    lst.write_text("".join(f"file '{c.name}'\n" for c in clips))
    silent = OUT / "silent.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent)])
    final = OUT / "final.mp4"
    total = SCENES * SCENE_SECONDS
    run(["ffmpeg", "-y", "-i", str(silent), "-stream_loop", "-1", "-i", str(music),
         "-t", str(total), "-af", f"afade=t=in:d=1,afade=t=out:st={total-2}:d=2,loudnorm=I=-16",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final)])
    return final


# ---------------------------------------------------------------- youtube
def upload(video, story, thumb):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    creds = Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"],
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=os.environ["YT_CLIENT_ID"], client_secret=os.environ["YT_CLIENT_SECRET"],
                        scopes=["https://www.googleapis.com/auth/youtube.upload"])
    yt = build("youtube", "v3", credentials=creds)
    desc = story["description_ar"] + "\n\n" + story["title_en"] + "\n\nAI-generated animation for kids."
    body = {"snippet": {"title": story["title_ar"][:95], "description": desc, "tags": story["tags"],
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


def main():
    story = retry(make_story); log("story:", story["title_en"])
    (OUT / "story.json").write_text(json.dumps(story, ensure_ascii=False, indent=2))
    imgs = make_images(story)
    music = hf_music("cheerful playful kids cartoon music, xylophone, ukulele, upbeat") or procedural_music()
    video = make_video(story, imgs, music); log("video ready", video)
    thumb = OUT / "thumb.jpg"; Image.open(imgs[1]).resize((1280, 720)).save(thumb, quality=90)
    if DRY_RUN:
        log("DRY_RUN: upload skipped"); return
    upload(video, story, thumb)


if __name__ == "__main__":
    try: main()
    except Exception as e:
        log("FAILED:", e); sys.exit(1)

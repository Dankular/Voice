#!/usr/bin/env python3
"""Dump ElevenLabs Voice Library entries (name, description, preview_url).

Uses the official GET /v1/shared-voices endpoint. API key is read from the
ELEVENLABS_API_KEY environment variable (never hardcode it).

Examples:
  ELEVENLABS_API_KEY=... ./get_voices.py --use-case social_media -o voices.csv
  ./get_voices.py --search narrator --max 200 --format json -o narrators.json

Local OmniVoice voice-design TTS (see README):
  ./get_voices.py tts --text "Hello" --gender female --pitch low --accent "british accent" -o out.wav
"""
import argparse, csv, json, os, sys, time
import urllib.error, urllib.parse, urllib.request

URL = "https://api.elevenlabs.io/v1/shared-voices"
FIELDS = ["voice_id", "name", "description", "preview_url"]


def fetch_page(key, params, retries=4):
    req = urllib.request.Request(f"{URL}?{urllib.parse.urlencode(params)}",
                                 headers={"xi-api-key": key})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise SystemExit(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise


def iter_voices(key, filters, page_size, limit):
    page, n = 0, 0
    while True:
        data = fetch_page(key, {**filters, "page_size": page_size, "page": page})
        for v in data.get("voices", []):
            yield {f: v.get(f) for f in FIELDS}
            n += 1
            if limit and n >= limit:
                return
        if not data.get("has_more") or not data.get("voices"):
            return
        page += 1


def iter_saved(key, voice_type, page_size, limit):
    """Voices in the account's own list via GET /v2/voices (token-paginated)."""
    token, n = None, 0
    while True:
        params = {"page_size": page_size, "voice_type": voice_type}
        if token:
            params["next_page_token"] = token
        req = urllib.request.Request("https://api.elevenlabs.io/v2/voices?" + urllib.parse.urlencode(params),
                                     headers={"xi-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        for v in data.get("voices", []):
            yield {f: v.get(f) for f in FIELDS}
            n += 1
            if limit and n >= limit:
                return
        token = data.get("next_page_token")
        if not data.get("has_more") or not token:
            return


def tts_main(argv):
    """`get_voices.py tts ...` - local OmniVoice (ONNX) voice-design TTS."""
    from omnivoice_tts.attributes import FACETS, build_instruct

    ap = argparse.ArgumentParser(prog="get_voices.py tts",
                                 description="Synthesise speech with OmniVoice voice design. "
                                             "Attribute choices come from the upstream-documented vocabulary.")
    ap.add_argument("--text", help="text to speak")
    ap.add_argument("--gender", choices=FACETS["gender"])
    ap.add_argument("--age", choices=FACETS["age"])
    ap.add_argument("--pitch", choices=FACETS["pitch"], help="the closest documented control to 'tone'")
    ap.add_argument("--whisper", action="store_true")
    ap.add_argument("--accent", choices=FACETS["accent"], help="English speech only")
    ap.add_argument("--dialect", choices=FACETS["dialect"], help="Chinese speech only")
    ap.add_argument("--instruct", help="raw instruct string (overrides the attribute flags)")
    ap.add_argument("--language", help="language id or name, e.g. en / English (default: auto)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--model-dir", help="dir with the int4 backbone (default: download to cache)")
    ap.add_argument("--higgs-dir", help="dir with higgs_decoder.onnx (default: download to cache)")
    ap.add_argument("--list-attributes", action="store_true")
    ap.add_argument("-o", "--output", default="out.wav")
    a = ap.parse_args(argv)

    if a.list_attributes:
        for k, v in FACETS.items():
            print(f"{k}: {', '.join(v)}")
        return
    if not a.text:
        ap.error("--text is required")
    instruct = a.instruct or build_instruct(a.gender, a.age, a.pitch, "whisper" if a.whisper else None,
                                            a.accent, a.dialect)
    from omnivoice_tts.engine import SAMPLE_RATE, OmniVoiceONNX
    from omnivoice_tts.models import ensure_models
    import soundfile as sf
    model_dir, higgs_dir = (a.model_dir, a.higgs_dir) if a.model_dir and a.higgs_dir else ensure_models()
    wav = OmniVoiceONNX(model_dir, higgs_dir).synthesize(
        a.text, instruct, a.language, speed=a.speed, seed=a.seed)
    sf.write(a.output, wav, SAMPLE_RATE)
    print(f"wrote {a.output} ({len(wav) / SAMPLE_RATE:.2f}s) instruct={instruct!r}", file=sys.stderr)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "tts":
        return tts_main(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--use-case", help="e.g. social_media, narration, characters_animation, conversational")
    ap.add_argument("--category", help="e.g. professional, famous, high_quality")
    ap.add_argument("--language")
    ap.add_argument("--gender")
    ap.add_argument("--accent")
    ap.add_argument("--age")
    ap.add_argument("--search")
    ap.add_argument("--sort", choices=["trending", "created_date", "usage_character_count_1y", "cloned_by_count"])
    ap.add_argument("--saved", nargs="?", const="saved", metavar="TYPE",
                    help="list the account's own voices (/v2/voices) instead of the public library; "
                         "TYPE is the voice_type filter, default 'saved'")
    ap.add_argument("--max", type=int, default=0, help="stop after N voices (0 = all)")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--format", choices=["csv", "json", "jsonl"], default="csv")
    ap.add_argument("-o", "--output", default="-")
    a = ap.parse_args()

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("Set ELEVENLABS_API_KEY")
    filters = {k: v for k, v in {
        "use_cases": a.use_case, "category": a.category, "language": a.language,
        "gender": a.gender, "sort": a.sort, "accent": a.accent, "age": a.age, "search": a.search,
    }.items() if v}

    out = sys.stdout if a.output == "-" else open(a.output, "w", newline="", encoding="utf-8")
    rows = (iter_saved(key, a.saved, a.page_size, a.max) if a.saved
            else iter_voices(key, filters, a.page_size, a.max))
    count = 0
    if a.format == "csv":
        w = csv.DictWriter(out, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r); count += 1
    elif a.format == "jsonl":
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n"); count += 1
    else:
        lst = list(rows); count = len(lst)
        json.dump(lst, out, ensure_ascii=False, indent=2)
    if out is not sys.stdout:
        out.close()
    print(f"wrote {count} voices", file=sys.stderr)


if __name__ == "__main__":
    main()

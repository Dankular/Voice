"""Split big files into <= N MB parts (GitHub rejects files > 100 MB) and write manifest.json for the browser loader.
usage: split_parts.py SRC_DIR DST_DIR [part_mb=80] file1 file2 ..."""
import hashlib, json, os, sys
src, dst, mb = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sys.argv[4:]
os.makedirs(dst, exist_ok=True)
man = {"files": {}}
for name in files:
    data = open(os.path.join(src, name), "rb").read()
    n = max(1, -(-len(data) // (mb * 1024 * 1024)))
    step = -(-len(data) // n)                                   # near-equal parts
    parts = []
    for i in range(n):
        pn = f"{name}.part{i}" if n > 1 else name
        open(os.path.join(dst, pn), "wb").write(data[i * step:(i + 1) * step]); parts.append(pn)
    man["files"][name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": parts}
    print(name, len(data), "->", len(parts), "part(s)")
json.dump(man, open(os.path.join(dst, "manifest.json"), "w"), indent=1)

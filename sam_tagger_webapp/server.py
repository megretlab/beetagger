#!/usr/bin/env python3
"""Bee Tagger - standalone, stdlib-only server.

    python3 server.py [--port 8002] [--host 127.0.0.1]

Open a tracks CSV from the UI; images are loaded from its `crop_filepath` column
(falling back to <csv dir>/crops/<crop_filename>). Source CSVs are never modified;
labels autosave to labels/<dataset>.json.
"""
import argparse, csv, io, json, mimetypes, os, re, shutil, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
LABELS = os.environ.get("BEETAGGER_LABELS") or os.path.join(HERE, "labels")
TAGS_FILE = os.path.join(LABELS, "_tags.json")
DS_FILE = os.path.join(LABELS, "_datasets.json")
INFER_FILE = os.path.join(LABELS, "_infer.json")
CLASSES_FILE = os.path.join(LABELS, "_classes.json")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DEFAULT_TAGS = [
    {"id": "red", "name": "Red", "color": "#e5484d"},
    {"id": "orange", "name": "Orange", "color": "#f76b15"},
    {"id": "yellow", "name": "Yellow", "color": "#e5c100"},
    {"id": "green", "name": "Green", "color": "#30a46c"},
    {"id": "blue", "name": "Blue", "color": "#3e63dd"},
    {"id": "purple", "name": "Purple", "color": "#8e4ec6"},
    {"id": "pink", "name": "Pink", "color": "#e93d82"},
    {"id": "white", "name": "White", "color": "#f0f0f0"},
    {"id": "none", "name": "No tag", "color": "#8b8d98"},
]
# Label classes: what can be tagged, and the CSV column each one is exported to.
# "c" (choice, options = the tag palette above) and "n" (number) are built in; extra classes are
# {"id", "name", "column", "type": "choice"|"number", "options": [{id,name,color}] | "min"/"max"}.
DEFAULT_CLASSES = [
    {"id": "c", "name": "Color", "column": "tag_color", "type": "choice", "builtin": True, "enabled": True},
    {"id": "n", "name": "Number", "column": "tag_number", "type": "number", "builtin": True, "enabled": True, "min": 1, "max": 100},
]
RESERVED_COLS = {"track_key_sam", "tag_rotation", "pred_color", "pred_color_conf", "pred_number", "pred_number_conf",
                 "crop_filepath", "crop_filename"}
csv.field_size_limit(sys.maxsize)
lock = threading.Lock()
cache = {}          # name -> loaded dataset
TRACK_RE = re.compile(r"^(.*?)\.?(T\d+)_F(\d+)")


# ---------- helpers ----------
def read_json(p, default):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return default


def write_json(p, obj):
    os.makedirs(LABELS, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, p)


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name)


def label_path(name): return os.path.join(LABELS, safe(name) + ".json")
def pred_path(name): return os.path.join(LABELS, safe(name) + ".pred.json")
def get_tags(): return read_json(TAGS_FILE, DEFAULT_TAGS)
def get_datasets(): return read_json(DS_FILE, {})
def get_classes(): return read_json(CLASSES_FILE, None) or DEFAULT_CLASSES


def clean_classes(lst):
    """Validate the class list sent by the UI; built-in classes are always kept."""
    out, ids, cols = [], set(), set()
    for c in lst:
        cid, col = str(c.get("id") or "").strip(), str(c.get("column") or "").strip()
        if not cid or cid in ids:
            raise ValueError("bad or duplicate class id")
        if not col:
            raise ValueError("every class needs an output column name")
        if col in RESERVED_COLS or col.endswith("_source") or col in cols:
            raise ValueError(f"output column '{col}' is reserved or used twice")
        ids.add(cid); cols.add(col)
        e = {"id": cid, "name": str(c.get("name") or col).strip(), "column": col,
             "type": "number" if c.get("type") == "number" else "choice", "enabled": c.get("enabled") is not False}
        if cid in ("c", "n"):
            e["builtin"] = True
        if e["type"] == "number":
            try: lo, hi = float(c.get("min", 0)), float(c.get("max", 100))
            except (TypeError, ValueError): raise ValueError("min and max must be numbers")
            e["min"], e["max"] = int(lo) if lo == int(lo) else lo, int(hi) if hi == int(hi) else hi
        elif cid != "c":
            e["options"] = [{"id": str(o["id"]), "name": str(o.get("name") or o["id"]), "color": o.get("color") or "#8b8d98"}
                            for o in c.get("options", [])]
        out.append(e)
    if not {"c", "n"} <= ids:
        raise ValueError("the built-in Color and Number classes cannot be removed")
    return out


def row_fields(r, use_tid=False):
    """-> (image id, source path, track key, frame).
    use_tid: rows added by Merge carry a remapped track_id; honour it in the track key."""
    path = (r.get("crop_filepath") or r.get("new_filepath") or "").strip()
    fn = (r.get("crop_filename") or "").strip() or os.path.basename(path)
    if not fn:
        return None
    m = TRACK_RE.match(fn)
    key = ((m.group(1) + "." if m.group(1) else "") + m.group(2)) if m else \
        (r.get("video_name", "") + "|" + (r.get("track_id") or "?"))
    if use_tid and m:
        try:
            tid = int(float(r.get("track_id")))
            if tid != int(m.group(2)[1:]):
                key = (m.group(1) + "." if m.group(1) else "") + "T" + str(tid).zfill(len(m.group(2)) - 1)
        except (TypeError, ValueError):
            pass
    try:
        frame = int(float(r.get("frame") or (m.group(3) if m else 0)))
    except ValueError:
        frame = 0
    return fn, path, key, frame


def norm_labels(lab):
    """Migrate v1 labels (plain tag-id strings) to {c, n} dicts."""
    for kind in ("tracks", "images"):
        d = lab.setdefault(kind, {})
        for k, v in list(d.items()):
            if isinstance(v, str):
                d[k] = {"c": v}
    return lab


def load_labels(name):
    return norm_labels(read_json(label_path(name), {"tracks": {}, "images": {}}))


# ---------- datasets ----------
def extra_path(name): return os.path.join(LABELS, safe(name) + ".extra.csv")


def iter_rows(name, csv_path):
    """Yield (row, use_tid) for the base CSV, then for rows appended by Merge."""
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            yield r, False
    if os.path.exists(extra_path(name)):
        with open(extra_path(name), newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                yield r, True


def stamp(name, csv_path):
    st = [os.stat(csv_path).st_mtime_ns]
    st.append(os.stat(extra_path(name)).st_mtime_ns if os.path.exists(extra_path(name)) else 0)
    return st


def load_dataset(name, csv_path):
    tracks, order, paths, maxid = {}, [], {}, {}
    for r, use_tid in iter_rows(name, csv_path):
        f = row_fields(r, use_tid)
        if not f:
            continue
        fn, path, key, frame = f
        paths[fn] = path
        t = tracks.get(key)
        if t is None:
            t = tracks[key] = {"key": key, "video": r.get("video_name", ""), "images": []}
            order.append(key)
        t["images"].append({"f": fn, "fr": frame})
        ids = []
        try: ids.append(int(float(r.get("track_id"))))
        except (TypeError, ValueError): pass
        m = TRACK_RE.match(fn)
        if m: ids.append(int(m.group(2)[1:]))
        if ids:
            v = r.get("video_name", "")
            maxid[v] = max(maxid.get(v, -1), *ids)
    for t in tracks.values():
        t["images"].sort(key=lambda i: i["fr"])
    return {"csv": csv_path, "stamp": stamp(name, csv_path), "tracks": tracks, "order": order, "paths": paths, "maxid": maxid}


def get_dataset(name):
    csv_path = get_datasets()[name]
    with lock:
        c = cache.get(name)
        if not c or c["stamp"] != stamp(name, csv_path):
            c = cache[name] = load_dataset(name, csv_path)
        return c


def resolve_path(ds, f):
    base = os.path.dirname(ds["csv"])
    p0 = ds["paths"].get(f)
    # older CSVs point at the original machine's crop folder; the same crops live under /mnt/data here
    moved = p0.replace("/home/reu_student_2023/Documents/backups/crops", "/mnt/data/bee_feeder/crops") if p0 else None
    for p in (p0, moved, os.path.join(base, "crops", f), os.path.join(base, f)):
        if p and os.path.isfile(p):
            return p
    return None


def export_csv(name, out_dir=None):
    """Write the tagged CSV on this machine (default: labels/); returns its path."""
    out_dir = os.path.abspath(os.path.expanduser(out_dir)) if out_dir else LABELS
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, tagged_name(name))
    with open(out, "w", newline="", encoding="utf-8") as oh:
        write_tagged_csv(name, oh)
    return out


def tagged_name(name): return safe(name) + ".tagged.csv"


def write_tagged_csv(name, oh):
    ds = get_dataset(name)
    lab = load_labels(name)
    tags = {t["id"]: t["name"] for t in get_tags()}
    preds = read_json(pred_path(name), {})
    classes = [c for c in get_classes() if c.get("enabled", True)]
    opts = {c["id"]: {o["id"]: o["name"] for o in c.get("options", [])} for c in classes}
    opts["c"] = tags
    new = ["track_key_sam"] + [x for c in classes for x in (c["column"], c["column"] + "_source")] + \
          ["tag_rotation", "pred_color", "pred_color_conf", "pred_number", "pred_number_conf"]
    cols = []
    for r, _ in iter_rows(name, ds["csv"]):        # header union (base columns first)
        for c in r:
            if c not in cols and c not in new:
                cols.append(c)
    w = csv.DictWriter(oh, fieldnames=cols + new, extrasaction="ignore")
    w.writeheader()
    for r, use_tid in iter_rows(name, ds["csv"]):
        f = row_fields(r, use_tid)
        if f:
            fn, _, key, _ = f
            il, tl = lab["images"].get(fn, {}), lab["tracks"].get(key, {})
            r["track_key_sam"] = key
            for c in classes:
                field = c["id"]
                if field in il:
                    v, src = il[field], "image"
                elif field in tl:
                    v, src = tl[field], "track"
                else:
                    v, src = "", ""
                r[c["column"]] = opts[field].get(v, v) if c["type"] == "choice" else v
                r[c["column"] + "_source"] = src
            r["tag_rotation"] = il.get("r", "")
            p = preds.get(fn)
            if p:
                r["pred_color"] = tags.get(p[0], p[0] or "")
                r["pred_color_conf"], r["pred_number"], r["pred_number_conf"] = p[1], p[2] if p[2] is not None else "", p[3]
        w.writerow(r)


# ---------- merge ----------
GUESS = {"c": ["tag_color", "color", "tag_colour"], "n": ["ground_truth_numbers", "ground_truth_number", "tag_number", "gt_number", "number"],
         "r": ["tag_rotation", "rotation", "tag_angle"]}


def guess_col(cols, kind):
    low = {c.lower(): c for c in cols}
    for g in GUESS[kind]:
        if g in low:
            return low[g]
    return ""


def read_source(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise ValueError("file not found: " + path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        return rd.fieldnames or [], rows


def path_map(ds):
    return {p: fn for fn, p in ds["paths"].items() if p}


def merge_scan(name, path):
    cols, rows = read_source(path)
    if "crop_filepath" not in cols:
        raise ValueError("the source CSV has no crop_filepath column (needed to match rows)")
    pm = path_map(get_dataset(name))
    paths = [(r.get("crop_filepath") or "").strip() for r in rows]
    matched = sum(1 for p in paths if p in pm)
    return {"columns": cols, "rows": len(rows), "matched": matched, "missing": len(rows) - matched,
            "guess": {k: guess_col(cols, k) for k in "cnr"}}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "tag"


def merge_run(name, b):
    cols, rows = read_source(b["path"])
    if "crop_filepath" not in cols:
        raise ValueError("the source CSV has no crop_filepath column")
    use = {k: bool(b.get("copy_" + k)) for k in "cnr"}
    colname = {k: (b.get("col_" + k) or "").strip() for k in "cnr"}
    for k in "cnr":
        if use[k] and colname[k] not in cols:
            raise ValueError(f"column '{colname[k]}' not found in the source CSV — enter the right column name")
    ds = get_dataset(name)
    pm = path_map(ds)
    res = {"source_rows": len(rows), "matched": 0, "added": 0, "tracks_renumbered": 0, "colors": 0, "numbers": 0, "rotations": 0,
           "kept_existing": 0, "bad_number": 0, "new_tags": [], "skipped_dup_filename": 0, "not_in_csv": 0}
    src_path = lambda r: (r.get("crop_filepath") or "").strip()
    res["matched"] = sum(1 for r in rows if src_path(r) in pm)
    missing = [r for r in rows if src_path(r) and src_path(r) not in pm]
    res["not_in_csv"] = len(missing)
    dry = bool(b.get("dry"))

    # ---- backup (for undo) ----
    if not dry:
        bdir = os.path.join(LABELS, "backups", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(bdir, exist_ok=True)
        info = {"name": name, "dir": bdir, "files": {}}
        for tag, p in (("labels", label_path(name)), ("extra", extra_path(name)), ("tags", TAGS_FILE)):
            info["files"][tag] = p if os.path.exists(p) else None
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(bdir, tag))
        write_json(os.path.join(LABELS, "_lastmerge_" + safe(name) + ".json"), info)

    # ---- add rows that are not in the working CSV, with unique track ids per video ----
    if b.get("add_missing") and missing:
        fn2key = {im["f"]: k for k, t in ds["tracks"].items() for im in t["images"]}
        cur = dict(ds["maxid"]); newid = {}; seen = set(); add = []
        def sid_of(r):
            try: return int(float(r.get("track_id")))
            except (TypeError, ValueError):
                m = TRACK_RE.match((r.get("crop_filename") or os.path.basename(src_path(r))))
                return int(m.group(2)[1:]) if m else 0
        adopt = {}      # (video, src tid) -> existing target tid for source tracks that already partly exist
        for r in rows:
            p = src_path(r)
            if p in pm and pm[p] in fn2key:
                m = re.search(r"T(\d+)$", fn2key[pm[p]])
                if m:
                    adopt.setdefault((r.get("video_name", ""), sid_of(r)), int(m.group(1)))
        for r in missing:
            fn = (r.get("crop_filename") or "").strip() or os.path.basename(src_path(r))
            if fn in ds["paths"] or fn in seen:
                res["skipped_dup_filename"] += 1; continue
            seen.add(fn)
            g = (r.get("video_name", ""), sid_of(r))
            if g not in newid:
                if g in adopt:
                    newid[g] = adopt[g]
                else:
                    cur[g[0]] = cur.get(g[0], -1) + 1
                    newid[g] = cur[g[0]]; res["tracks_renumbered"] += 1
            r = dict(r); r["track_id"] = str(newid[g]); add.append(r)
        res["added"] = len(add)
        if add and not dry:
            old = []
            if os.path.exists(extra_path(name)):
                with open(extra_path(name), newline="", encoding="utf-8-sig") as fh:
                    old = list(csv.DictReader(fh))
            allc = []
            for r in old + add:
                for c in r:
                    if c not in allc: allc.append(c)
            with open(extra_path(name), "w", newline="", encoding="utf-8") as oh:
                w = csv.DictWriter(oh, fieldnames=allc, restval="")
                w.writeheader(); w.writerows(old + add)
            ds = get_dataset(name)
            pm = path_map(ds)

    # ---- copy color / number / rotation ----
    tags = get_tags(); tmap = {}
    for t in tags:
        tmap[t["id"].lower()] = t["id"]; tmap[t["name"].lower()] = t["id"]
    def color_id(v):
        v = (v or "").strip()
        if not v or v.lower() in ("nan", "null"):
            return None
        if v.lower() not in tmap:
            base = slug(v); i, nid = 2, base
            while nid in {t["id"] for t in tags}:
                nid = f"{base}_{i}"; i += 1
            hue = sum(map(ord, v)) * 47 % 360
            tags.append({"id": nid, "name": v, "color": hsl_hex(hue)}); res["new_tags"].append(v)
            tmap[v.lower()] = nid
        return tmap[v.lower()]
    sv = {}
    for r in rows:
        p = src_path(r)
        if not p: continue
        e = {}
        if use["c"]:
            c = color_id(r.get(colname["c"]))
            if c: e["c"] = c
        if use["n"]:
            raw = (r.get(colname["n"]) or "").strip()
            if raw and raw.lower() not in ("nan", "null"):
                try:
                    n = int(round(float(raw)))
                    if 1 <= n <= 100: e["n"] = n
                    else: res["bad_number"] += 1
                except ValueError:
                    res["bad_number"] += 1
        if use["r"]:
            raw = (r.get(colname["r"]) or "").strip()
            try: e["r"] = round(float(raw) % 360, 2)
            except ValueError: pass
        if e: sv[p] = e
    lab = load_labels(name)
    overwrite = bool(b.get("overwrite"))
    def put(d, k, fld, v):
        d.setdefault(k, {})[fld] = v
    def drop(d, k, fld):
        if k in d:
            d[k].pop(fld, None)
            if not d[k]: del d[k]
    for k, t in ds["tracks"].items():
        imgs = [im["f"] for im in t["images"]]
        per = {f: sv[ds["paths"][f]] for f in imgs if ds["paths"].get(f) in sv}
        if not per:
            continue
        for fld, stat in (("c", "colors"), ("n", "numbers"), ("r", "rotations")):
            vals = {f: e[fld] for f, e in per.items() if fld in e}
            if not vals:
                continue
            if fld == "r":
                for f, v in vals.items():
                    if fld in lab["images"].get(f, {}) and not overwrite: res["kept_existing"] += 1; continue
                    put(lab["images"], f, "r", v); res[stat] += 1
                continue
            tl = lab["tracks"].get(k, {})
            if len(vals) == len(imgs) and len(set(vals.values())) == 1:
                v = next(iter(vals.values()))
                if fld in tl and not overwrite:
                    res["kept_existing"] += len(imgs)
                else:
                    put(lab["tracks"], k, fld, v); res[stat] += len(imgs)
                    if overwrite:
                        for f in imgs: drop(lab["images"], f, fld)
            else:
                for f, v in vals.items():
                    tl = lab["tracks"].get(k, {})
                    existing = lab["images"].get(f, {}).get(fld, tl.get(fld))
                    if existing is not None and not overwrite:
                        res["kept_existing"] += 1; continue
                    if tl.get(fld) == v: drop(lab["images"], f, fld)
                    else: put(lab["images"], f, fld, v)
                    res[stat] += 1
    if not dry:
        write_json(TAGS_FILE, tags)
        write_json(label_path(name), lab)
    else:
        res["new_tags"] = res["new_tags"]
    return res


def hsl_hex(h, s=0.65, l=0.55):
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def merge_undo(name):
    p = os.path.join(LABELS, "_lastmerge_" + safe(name) + ".json")
    info = read_json(p, None)
    if not info:
        raise ValueError("no merge to undo")
    for tag, target in (("labels", label_path(name)), ("extra", extra_path(name)), ("tags", TAGS_FILE)):
        bak = os.path.join(info["dir"], tag)
        if info["files"].get(tag) and os.path.exists(bak):
            shutil.copy2(bak, target)
        elif os.path.exists(target):
            os.remove(target)
    os.remove(p)
    return {"restored": info["dir"]}


# ---------- inference job ----------
job = {"state": "idle", "done": 0, "total": 0, "msg": "", "name": None}
job_stop = threading.Event()


def run_infer(name, cfg):
    global job
    try:
        ds = get_dataset(name)
        lab = load_labels(name)
        tags = get_tags()
        tmap = {}
        for t in tags:
            tmap[t["id"].lower()] = t["id"]; tmap[t["name"].lower()] = t["id"]
        per = int(cfg.get("per_track") or 0)
        items = []
        for k in ds["order"]:
            tl = lab["tracks"].get(k, {})
            if cfg.get("scope") == "unlabeled" and "c" in tl and "n" in tl:
                continue
            ims = ds["tracks"][k]["images"]
            if per and len(ims) > per:
                ims = [ims[int(i * len(ims) / per)] for i in range(per)]
            items += [i["f"] for i in ims]
        paths = [(f, resolve_path(ds, f)) for f in items]
        missing = sum(1 for _, p in paths if not p)
        paths = [(f, p) for f, p in paths if p]
        job.update(state="running", done=0, total=len(paths), msg=f"{missing} images not found" if missing else "", name=name)
        if not paths:
            raise RuntimeError("no readable images to run on")
        log = open(os.path.join(LABELS, "_infer.log"), "w")
        py = cfg.get("python") or sys.executable
        proc = subprocess.Popen([py, os.path.join(HERE, "runner.py"), cfg["script"], cfg.get("weights") or ""],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)

        def readmsg():
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("model process exited:\n" + open(os.path.join(LABELS, "_infer.log")).read()[-1500:])
            m = json.loads(line)
            if "error" in m:
                raise RuntimeError(m["error"][-1500:])
            return m

        readmsg()   # ready
        preds = read_json(pred_path(name), {}) if cfg.get("scope") == "unlabeled" else {}
        bs = max(1, int(cfg.get("batch") or 32))
        for bi in range(0, len(paths), bs):
            if job_stop.is_set():
                job["msg"] = "stopped"; break
            chunk = paths[bi:bi + bs]
            proc.stdin.write(json.dumps({"id": bi, "paths": [p for _, p in chunk]}) + "\n"); proc.stdin.flush()
            res = readmsg()["results"]
            for (f, _), r in zip(chunk, res):
                c = r.get("color")
                c = tmap.get(str(c).lower()) if c is not None else None
                n = r.get("number")
                try:
                    n = int(round(float(n)))
                except (TypeError, ValueError):
                    n = None
                if n is not None and not 1 <= n <= 100:
                    n = None
                preds[f] = [c, float(r.get("color_conf") or 0), n, float(r.get("number_conf") or 0)]
            job["done"] = bi + len(chunk)
            if (bi // bs) % 25 == 24:
                write_json(pred_path(name), preds)
        write_json(pred_path(name), preds)
        try:
            proc.stdin.close(); proc.terminate()
        except Exception:
            pass
        job["state"] = "done"
    except Exception as e:
        job.update(state="error", msg=str(e))


# ---------- http ----------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json", extra=None):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        p = unquote(u.path)
        try:
            if p == "/api/datasets":
                return self.send(200, [{"name": n, "csv": c} for n, c in get_datasets().items()])
            if p == "/api/tags":
                return self.send(200, get_tags())
            if p == "/api/classes":
                return self.send(200, get_classes())
            if p == "/api/browse":
                d = os.path.abspath(os.path.expanduser(q.get("path") or "~"))
                if not os.path.isdir(d):
                    d = os.path.dirname(d)
                ents = sorted(os.scandir(d), key=lambda e: e.name.lower())
                return self.send(200, {"path": d, "parent": os.path.dirname(d),
                    "dirs": [e.name for e in ents if e.is_dir() and not e.name.startswith(".")],
                    "files": [e.name for e in ents if e.is_file() and e.name.lower().endswith(".csv")]})
            if p == "/api/dataset":
                ds = get_dataset(q["name"])
                sample = [t["images"][0]["f"] for t in list(ds["tracks"].values())[:5]]
                ok = sum(1 for f in sample if resolve_path(ds, f))
                return self.send(200, {"tracks": [ds["tracks"][k] for k in ds["order"]],
                                       "labels": load_labels(q["name"]), "probe": [ok, len(sample)]})
            if p == "/api/preds":
                return self.send(200, read_json(pred_path(q["name"]), {}))
            if p == "/api/infer/status":
                return self.send(200, job)
            if p == "/api/infer/config":
                return self.send(200, read_json(INFER_FILE, {"python": sys.executable,
                                "script": os.path.join(HERE, "models", "example_model.py"), "weights": "",
                                "batch": 32, "per_track": 8, "scope": "all"}))
            if p == "/api/img":
                ds = get_dataset(q["name"])
                fp = resolve_path(ds, q["f"])
                if not fp or not fp.lower().endswith(IMG_EXT):
                    return self.send(404, b"", "text/plain")
                with open(fp, "rb") as fh:
                    return self.send(200, fh.read(), mimetypes.guess_type(fp)[0] or "image/jpeg",
                                     {"Cache-Control": "max-age=86400"})
            rel = "index.html" if p == "/" else p.lstrip("/")
            fp = os.path.normpath(os.path.join(STATIC, rel))
            if fp.startswith(STATIC) and os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    return self.send(200, fh.read(), mimetypes.guess_type(fp)[0] or "text/plain", {"Cache-Control": "no-cache"})
            self.send(404, b"not found", "text/plain")
        except Exception as e:
            self.send(500, {"error": repr(e)})

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            b = self.body()
            if p == "/api/open":
                path = os.path.abspath(os.path.expanduser(b["path"]))
                if not os.path.isfile(path):
                    return self.send(400, {"error": "file not found: " + path})
                dsets = get_datasets()
                name = next((n for n, c in dsets.items() if c == path), None)
                if not name:
                    name = base = os.path.splitext(os.path.basename(path))[0]
                    i = 2
                    while name in dsets:
                        name = f"{base}_{i}"; i += 1
                    dsets[name] = path
                    write_json(DS_FILE, dsets)
                get_dataset(name)
                return self.send(200, {"name": name})
            if p == "/api/forget":
                d = get_datasets(); d.pop(b["name"], None); write_json(DS_FILE, d)
                return self.send(200, {"ok": True})
            if p == "/api/tags":
                write_json(TAGS_FILE, b["tags"])
                return self.send(200, {"ok": True})
            if p == "/api/classes":
                try: cl = clean_classes(b["classes"])
                except ValueError as e: return self.send(400, {"error": str(e)})
                write_json(CLASSES_FILE, cl)
                return self.send(200, cl)
            if p == "/api/labels":
                # patch: {name, tracks:{key:{c?:id|null, n?:int|null}}, images:{...}}
                with lock:
                    lp = label_path(b["name"])
                    lab = load_labels(b["name"])
                    for kind in ("tracks", "images"):
                        for k, patch in (b.get(kind) or {}).items():
                            e = lab[kind].setdefault(k, {})
                            for field, v in patch.items():
                                if v is None:
                                    e.pop(field, None)
                                else:
                                    e[field] = v
                            if not e:
                                del lab[kind][k]
                    write_json(lp, lab)
                return self.send(200, {"ok": True})
            if p == "/api/merge/scan":
                try: return self.send(200, merge_scan(b["name"], b["path"]))
                except ValueError as e: return self.send(400, {"error": str(e)})
            if p == "/api/merge/run":
                try: return self.send(200, merge_run(b["name"], b))
                except ValueError as e: return self.send(400, {"error": str(e)})
            if p == "/api/merge/undo":
                try: return self.send(200, merge_undo(b["name"]))
                except ValueError as e: return self.send(400, {"error": str(e)})
            if p == "/api/export":
                if b.get("mode") == "download":     # send the CSV to the browser (local machine)
                    buf = io.StringIO(newline="")
                    write_tagged_csv(b["name"], buf)
                    return self.send(200, buf.getvalue().encode("utf-8"), "text/csv",
                                     {"Content-Disposition": 'attachment; filename="%s"' % tagged_name(b["name"])})
                return self.send(200, {"path": export_csv(b["name"], b.get("dir"))})
            if p == "/api/purge_tag":
                n = 0
                for name in get_datasets():
                    if not os.path.exists(label_path(name)):
                        continue
                    lab = load_labels(name)
                    for kind in ("tracks", "images"):
                        for k in list(lab[kind]):
                            if lab[kind][k].get("c") == b["id"]:
                                del lab[kind][k]["c"]; n += 1
                                if not lab[kind][k]:
                                    del lab[kind][k]
                    write_json(label_path(name), lab)
                return self.send(200, {"removed": n})
            if p == "/api/infer/start":
                if job["state"] == "running":
                    return self.send(400, {"error": "a job is already running"})
                write_json(INFER_FILE, b["config"])
                job_stop.clear()
                job.update(state="running", done=0, total=0, msg="starting…", name=b["name"])
                threading.Thread(target=run_infer, args=(b["name"], b["config"]), daemon=True).start()
                return self.send(200, {"ok": True})
            if p == "/api/infer/stop":
                job_stop.set()
                return self.send(200, {"ok": True})
            if p == "/api/infer/clear":
                if os.path.exists(pred_path(b["name"])):
                    os.remove(pred_path(b["name"]))
                return self.send(200, {"ok": True})
            self.send(404, {"error": "unknown"})
        except Exception as e:
            self.send(500, {"error": repr(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    os.makedirs(LABELS, exist_ok=True)
    print(f"Bee Tagger: http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()

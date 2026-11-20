#!/usr/bin/env python3
"""qrstream - PDF (or any file) <-> tiled QR-code video, using only OpenCV + Python.

    python qrstream.py encode    input.pdf  out.avi  --resolution 1920x1080 --grid 4x2 --fec 20
    python qrstream.py decode    out.avi    recovered.pdf
    python qrstream.py benchmark input.pdf
    python qrstream.py codecs    [--resolution 1920x1080]

Pipeline
    file -> SHA-256 -> zstd (only if smaller) -> fixed-size binary chunks
          -> block erasure code (Cauchy Reed-Solomon over GF(256), across packets)
          -> packets (28-byte binary header, CRC32) -> whitening -> QR (byte mode, raw bytes)
          -> several QR symbols tiled per frame -> cv2.VideoWriter (lossless codec if available)

No FFmpeg executable is needed (OpenCV's bundled backend does the video I/O), nothing is
Base64-encoded, and the whole thing works offline.  Raw bytes survive the QR round trip
because OpenCV hands back byte-mode payloads as Latin-1 text re-encoded as UTF-8, which is
exactly reversible (see `_fix_decoded`).
"""
import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import argparse
import hashlib
import math
import random
import struct
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import cv2
import numpy as np
import qrcode
from qrcode import base as qr_base
from qrcode import util as qr_util
from qrcode.constants import ERROR_CORRECT_H, ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q

try:
    import zstandard as zstd
except ImportError:  # compression is optional on the encode side
    zstd = None

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Packet format
# ---------------------------------------------------------------------------
MAGIC = b"QRSV"
PROTO_VERSION = 1
KIND_META, KIND_CODED = 0, 1
# magic(4) version(1) kind(1) file_id(8) packet_id(4) total(4) payload_len(2) crc32(4)
HDR = struct.Struct(">4sBB8sIIHI")
HDR_LEN = HDR.size  # 28
# proto comp flags name_len | orig stored sha256 | data_pkts total_pkts chunk fec_pct k_max
# | cols rows | W H | version ecc px quiet | crc32(stored stream)
META = struct.Struct(">BBBBQQ32sIIHHHBBHHBBBBI")
COMP_NONE, COMP_ZSTD = 0, 1
COMP_NAMES = {COMP_NONE: "none", COMP_ZSTD: "zstd"}
ECC_NAMES = "LMQH"
ECC_QR = {"L": ERROR_CORRECT_L, "M": ERROR_CORRECT_M, "Q": ERROR_CORRECT_Q, "H": ERROR_CORRECT_H}
QR_MASK = 2  # fixed mask pattern: ~8x faster than scoring all 8; payload is whitened so it is safe

# Fixed whitening keystream (deterministic, portable): avoids long runs of identical modules
_KEY = b"".join(hashlib.sha256(b"qrstream-whiten" + i.to_bytes(4, "big")).digest() for i in range(128))
_KEY_NP = np.frombuffer(_KEY, dtype=np.uint8)


def whiten(b: bytes) -> bytes:
    return (np.frombuffer(b, dtype=np.uint8) ^ _KEY_NP[: len(b)]).tobytes()


def pack_packet(kind: int, file_id: bytes, pid: int, total: int, payload: bytes) -> bytes:
    hdr0 = HDR.pack(MAGIC, PROTO_VERSION, kind, file_id, pid, total, len(payload), 0)
    crc = zlib.crc32(payload, zlib.crc32(hdr0)) & 0xFFFFFFFF
    return HDR.pack(MAGIC, PROTO_VERSION, kind, file_id, pid, total, len(payload), crc) + payload


def unpack_packet(raw: bytes):
    """-> (kind, file_id, pid, total, payload) or None if magic/version/length/CRC is bad."""
    if len(raw) < HDR_LEN:
        return None
    magic, ver, kind, fid, pid, total, plen, crc = HDR.unpack_from(raw)
    if magic != MAGIC or ver != PROTO_VERSION or HDR_LEN + plen > len(raw):
        return None
    payload = raw[HDR_LEN : HDR_LEN + plen]
    hdr0 = HDR.pack(magic, ver, kind, fid, pid, total, plen, 0)
    if zlib.crc32(payload, zlib.crc32(hdr0)) & 0xFFFFFFFF != crc:
        return None
    return kind, fid, pid, total, payload


# ---------------------------------------------------------------------------
# Erasure code: systematic Cauchy Reed-Solomon over GF(256), vectorised with numpy
# ---------------------------------------------------------------------------
def _gf_tables():
    exp = np.zeros(512, dtype=np.int64)
    log = np.zeros(256, dtype=np.int64)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    exp[255:510] = exp[:255]
    a = np.arange(1, 256)
    mul = np.zeros((256, 256), dtype=np.uint8)
    mul[1:, 1:] = exp[log[a][:, None] + log[a][None, :]]
    inv = np.zeros(256, dtype=np.uint8)
    inv[1:] = exp[255 - log[a]]
    return mul, inv


GF_MUL, GF_INV = _gf_tables()


def _cauchy(k: int, m: int) -> np.ndarray:
    """m x k matrix; every square submatrix is invertible, so any k of k+m packets suffice."""
    x = np.arange(m)
    y = np.arange(m, m + k)
    return GF_INV[x[:, None] ^ y[None, :]]


def fec_encode(rows: np.ndarray, m: int) -> np.ndarray:
    k, length = rows.shape
    c = _cauchy(k, m)
    par = np.zeros((m, length), dtype=np.uint8)
    for i in range(k):
        par ^= GF_MUL[c[:, i][:, None], rows[i][None, :]]
    return par


def fec_decode(k: int, m: int, data: dict, parity: dict, length: int):
    """data: {i: uint8[length]}, parity: {j: uint8[length]} -> k x length matrix, or None."""
    missing = [i for i in range(k) if i not in data]
    out = np.zeros((k, length), dtype=np.uint8)
    for i, v in data.items():
        out[i] = v
    e = len(missing)
    if e == 0:
        return out
    use = sorted(parity)[:e]
    if len(use) < e:
        return None
    c = _cauchy(k, m)
    rhs = np.stack([parity[j] for j in use]).astype(np.uint8)
    for i, v in data.items():
        rhs ^= GF_MUL[c[use, i][:, None], v[None, :]]
    a = c[np.ix_(use, missing)].copy()
    for col in range(e):
        piv = next((r for r in range(col, e) if a[r, col]), None)
        if piv is None:
            return None
        if piv != col:
            a[[col, piv]] = a[[piv, col]]
            rhs[[col, piv]] = rhs[[piv, col]]
        inv = GF_INV[a[col, col]]
        a[col] = GF_MUL[inv][a[col]]
        rhs[col] = GF_MUL[inv][rhs[col]]
        for r in range(e):
            if r != col and a[r, col]:
                f = a[r, col]
                a[r] ^= GF_MUL[f][a[col]]
                rhs[r] ^= GF_MUL[f][rhs[col]]
    for n, i in enumerate(missing):
        out[i] = rhs[n]
    return out


def choose_kmax(pct: int) -> int:
    k = 255
    while k > 1 and k + -(-k * pct // 100) > 255:
        k -= 1
    return k


def block_layout(d: int, kmax: int, pct: int):
    """Deterministic block structure -> ([(offset, k, m, first_data_index)], total_packets)."""
    nb = max(1, -(-d // kmax))
    base, extra = divmod(d, nb)
    blocks, off, did = [], 0, 0
    for b in range(nb):
        k = base + (1 if b < extra else 0)
        m = -(-k * pct // 100) if pct else 0
        blocks.append((off, k, m, did))
        off += k + m
        did += k
    return blocks, off


# ---------------------------------------------------------------------------
# QR helpers
# ---------------------------------------------------------------------------
def qr_capacity(version: int, ecc: str) -> int:
    """Max raw bytes in a byte-mode symbol."""
    blocks = qr_base.rs_blocks(version, ECC_QR[ecc])
    bits = 8 * sum(b.data_count for b in blocks)
    return (bits - 4 - (8 if version < 10 else 16)) // 8


def qr_modules(version: int) -> int:
    return 17 + 4 * version


def make_matrix(args) -> np.ndarray:
    """(raw_bytes, version, ecc) -> n x n uint8 (1 = dark).  Top-level so it can run in a pool."""
    data, version, ecc = args
    qr = qrcode.QRCode(version=version, error_correction=ECC_QR[ecc], border=0, box_size=1, mask_pattern=QR_MASK)
    qr.add_data(qr_util.QRData(data, mode=qr_util.MODE_8BIT_BYTE))
    qr.make(fit=False)
    return np.array(qr.get_matrix(), dtype=np.uint8)


def render_frame(mats, cfg: "Cfg") -> np.ndarray:
    frame = np.full((cfg.height, cfg.width), 255, dtype=np.uint8)
    cw, ch = cfg.width // cfg.cols, cfg.height // cfg.rows
    for idx, m in enumerate(mats):
        r, c = divmod(idx, cfg.cols)
        img = np.repeat(np.repeat((1 - m) * 255, cfg.px, 0), cfg.px, 1).astype(np.uint8)
        s = img.shape[0]
        y0, x0 = r * ch + (ch - s) // 2, c * cw + (cw - s) // 2
        frame[y0 : y0 + s, x0 : x0 + s] = img
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def _fix_decoded(b: bytes):
    """OpenCV returns byte-mode data as Latin-1 text encoded as UTF-8; undo that."""
    try:
        return b.decode("utf-8").encode("latin-1")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return None


# ---------------------------------------------------------------------------
# Configuration / layout
# ---------------------------------------------------------------------------
@dataclass
class Cfg:
    width: int = 1920
    height: int = 1080
    cols: int = 4
    rows: int = 2
    version: int = 0  # 0 = largest that fits at `px`
    ecc: str = "L"
    px: int = 3  # pixels per QR module
    quiet: int = 2  # quiet-zone modules guaranteed around every symbol
    fps: int = 30
    payload: int = 0  # bytes of file data per QR (0 = max for the version)
    fec: int = 0  # percent parity
    passes: int = 1  # full retransmissions (each in a different order)
    meta_every: int = 8  # frames between metadata refreshes
    codec: str = "auto"
    level: int = 19  # zstd level
    compress: bool = True
    workers: int = 0

    @property
    def cells(self):
        return self.cols * self.rows


def resolve(cfg: Cfg) -> Cfg:
    """Fill in version / payload and validate that the QRs physically fit."""
    cw, ch = cfg.width // cfg.cols, cfg.height // cfg.rows
    avail = min(cw, ch) // cfg.px  # modules (incl. quiet zones) that fit in one cell
    vmax = min(40, (avail - 2 * cfg.quiet - 17) // 4)
    v = cfg.version or vmax
    if v < 1 or v > vmax:
        raise SystemExit(
            f"QR version {v} does not fit: cell is {cw}x{ch}px, {cfg.px}px/module, quiet={cfg.quiet} "
            f"-> max version {vmax}. Use a smaller version/px/grid."
        )
    cap = qr_capacity(v, cfg.ecc) - HDR_LEN
    pl = cfg.payload or cap
    if pl > cap or pl < 1:
        raise SystemExit(f"payload {pl} exceeds capacity {cap} bytes for v{v}-{cfg.ecc} (header is {HDR_LEN} bytes)")
    return replace(cfg, version=v, payload=pl)


def max_fit_version(width, height, cols, rows, px, quiet) -> int:
    cw, ch = width // cols, height // rows
    return min(40, (min(cw, ch) // px - 2 * quiet - 17) // 4)


# ---------------------------------------------------------------------------
# Stream construction (encode side)
# ---------------------------------------------------------------------------
def build_stream(data: bytes, name: str, cfg: Cfg):
    """-> (meta_packet, ordered packet list, info dict).  Packet order = transmission order."""
    if not data:
        raise SystemExit("input file is empty")
    sha = hashlib.sha256(data).digest()
    fid = sha[:8]
    stored, comp = data, COMP_NONE
    if cfg.compress and zstd is not None:
        c = zstd.ZstdCompressor(level=cfg.level).compress(data)
        if len(c) < len(data):
            stored, comp = c, COMP_ZSTD
    s = cfg.payload
    d = -(-len(stored) // s)
    kmax = choose_kmax(cfg.fec) if cfg.fec else 255
    blocks, total = block_layout(d, kmax, cfg.fec)
    coded = [b""] * total
    for off, k, m, did in blocks:
        rows = np.zeros((k, s), dtype=np.uint8)
        for i in range(k):
            chunk = stored[(did + i) * s : (did + i + 1) * s]
            coded[off + i] = chunk
            rows[i, : len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
        if m:
            par = fec_encode(rows, m)
            for j in range(m):
                coded[off + k + j] = par[j].tobytes()
    pk = [pack_packet(KIND_CODED, fid, i, total, coded[i]) for i in range(total)]

    nb = name.encode("utf-8")[:200].decode("utf-8", "ignore").encode("utf-8")
    meta = META.pack(
        PROTO_VERSION, comp, 0, len(nb), len(data), len(stored), sha, d, total, s, cfg.fec, kmax,
        cfg.cols, cfg.rows, cfg.width, cfg.height, cfg.version, ECC_NAMES.index(cfg.ecc), cfg.px, cfg.quiet,
        zlib.crc32(stored) & 0xFFFFFFFF,
    ) + nb
    meta_pkt = pack_packet(KIND_META, fid, 0, total, meta)

    # transmission order: pass 0 = data interleaved across blocks, then parity interleaved;
    # later passes = different deterministic shuffles
    order = []
    for pass_no in range(cfg.passes):
        if pass_no == 0:
            ids = []
            for i in range(max(k for _, k, _, _ in blocks)):
                for off, k, m, _ in blocks:
                    if i < k:
                        ids.append(off + i)
            for j in range(max(m for _, _, m, _ in blocks)):
                for off, k, m, _ in blocks:
                    if j < m:
                        ids.append(off + k + j)
        else:
            ids = list(range(total))
            random.Random(int.from_bytes(fid, "big") ^ pass_no).shuffle(ids)
        order.extend(pk[i] for i in ids)

    info = dict(sha=sha, stored=len(stored), comp=comp, data_pkts=d, total=total, blocks=len(blocks),
                orig=len(data), name=name)
    return meta_pkt, order, info


def frame_plan(meta_pkt: bytes, order: list, cells: int, meta_every: int):
    """Yield a list of packets (<= cells) per frame; frame 0 is all metadata, then periodic refresh."""
    yield [meta_pkt] * cells
    i, f = 0, 1
    while i < len(order):
        take = cells
        frame = []
        refresh = bool(meta_every) and f % meta_every == 0
        if refresh:
            take -= 1
        frame += order[i : i + take]
        i += take
        if refresh:  # rotate the metadata cell so one bad screen region cannot suppress it every time
            frame.insert((f // meta_every) % (len(frame) + 1), meta_pkt)
        f += 1
        while len(frame) < cells:
            frame.append(meta_pkt)  # padding doubles as extra metadata redundancy
        yield frame


# ---------------------------------------------------------------------------
# Codec probing / video I/O
# ---------------------------------------------------------------------------
CODEC_CANDIDATES = ["FFV1", "ULRG", "HFYU", "LAGS", "I420", "MJPG", "XVID", "mp4v", "VP90"]
_probe_cache = {}


def _open_writer(path, fourcc, fps, size):
    for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
        try:
            w = cv2.VideoWriter(path, backend, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        except Exception:
            continue
        if w.isOpened():
            return w
        w.release()
    return None


def _open_capture(path):
    for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
        cap = cv2.VideoCapture(path, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def probe_codecs(width, height, ext=".avi", px=3, verbose=False):
    """Actually open each codec with cv2.VideoWriter, write noisy QR-like frames, read them back.
    Returns [{codec, exact, maxerr, bytes}] best first (exact-lossless in preference order)."""
    key = (width, height, ext, px)
    if key in _probe_cache:
        return _probe_cache[key]
    rng = np.random.default_rng(7)
    frames = []
    for _ in range(3):
        bits = rng.integers(0, 2, (height // px + 1, width // px + 1), dtype=np.uint8) * 255
        g = np.repeat(np.repeat(bits, px, 0), px, 1)[:height, :width]
        frames.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    res = []
    with tempfile.TemporaryDirectory() as td:
        for n, cc in enumerate(CODEC_CANDIDATES):
            path = os.path.join(td, f"p{n}{ext}")  # fresh file per codec: no stale-file false positives
            try:
                w = _open_writer(path, cc, 30, (width, height))
                if w is None:
                    continue
                for f in frames:
                    w.write(f)
                w.release()
                cap = _open_capture(path)
                if cap is None:
                    continue
                back = []
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    back.append(fr)
                cap.release()
                if len(back) != len(frames) or back[0].shape != frames[0].shape:
                    continue
                err = max(int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()) for a, b in zip(frames, back))
                res.append(dict(codec=cc, exact=err == 0, maxerr=err, bytes=os.path.getsize(path)))
            except Exception:
                continue
    res.sort(key=lambda r: (not r["exact"], r["maxerr"], CODEC_CANDIDATES.index(r["codec"])))
    _probe_cache[key] = res
    return res


def pick_codec(cfg: Cfg, ext: str) -> str:
    if cfg.codec != "auto":
        return cfg.codec
    res = probe_codecs(cfg.width, cfg.height, ext, cfg.px)
    if not res:
        raise SystemExit("no usable cv2.VideoWriter codec found for this container/resolution")
    return res[0]["codec"]


_pool = None


def get_pool(workers):
    global _pool
    if workers > 1 and _pool is None:
        _pool = ProcessPoolExecutor(max_workers=workers)
    return _pool


def default_workers():
    return max(1, min(8, (os.cpu_count() or 2) - 1))


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------
def encode_bytes(data: bytes, name: str, out: str, cfg: Cfg, quiet_log=False):
    cfg = resolve(cfg)
    ext = os.path.splitext(out)[1].lower() or ".avi"
    codec = pick_codec(cfg, ext)
    t0 = time.time()
    meta_pkt, order, info = build_stream(data, name, cfg)
    writer = _open_writer(out, codec, cfg.fps, (cfg.width, cfg.height))
    if writer is None:
        raise SystemExit(f"cv2.VideoWriter could not open codec {codec} for {out}")
    cap = qr_capacity(cfg.version, cfg.ecc)
    workers = cfg.workers or default_workers()
    pool = get_pool(workers)
    plan = list(frame_plan(meta_pkt, order, cfg.cells, cfg.meta_every))
    cache = {}
    win = max(1, workers * 2)
    for w0 in range(0, len(plan), win):
        chunk = plan[w0 : w0 + win]
        need = {p for fr in chunk for p in fr if p not in cache}
        need = list(need)
        # fill every symbol to full capacity with whitened filler: a mostly-padding QR (EC/11 pad
        # pattern) confuses the detector; payload_len in the header lets the decoder ignore the filler
        items = [(whiten(p.ljust(cap, b"\0")), cfg.version, cfg.ecc) for p in need]
        mats = pool.map(make_matrix, items, chunksize=2) if pool and len(items) > 8 else map(make_matrix, items)
        for p, m in zip(need, mats):
            cache[p] = m
        for fr in chunk:
            writer.write(render_frame([cache[p] for p in fr], cfg))
        if not quiet_log:
            print(f"\r  frames {min(w0 + win, len(plan))}/{len(plan)}", end="", file=sys.stderr, flush=True)
        if len(cache) > 4096:
            cache = {k: v for k, v in cache.items() if k == meta_pkt}
    writer.release()
    if not quiet_log:
        print(file=sys.stderr)
    nframes = len(plan)
    return dict(cfg=cfg, codec=codec, frames=nframes, info=info, enc_time=time.time() - t0,
                video_bytes=os.path.getsize(out), duration=nframes / cfg.fps)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
GUESS_GRIDS = [(4, 2), (3, 2), (2, 2), (2, 1), (1, 1), (3, 3), (4, 3), (5, 3), (6, 3), (4, 4), (5, 2), (6, 2),
               (3, 1), (4, 1), (1, 2), (1, 3), (2, 3), (5, 4), (6, 4), (8, 4), (8, 6)]


class FileState:
    def __init__(self):
        self.meta = None
        self.pk = {}


def parse_meta(payload: bytes):
    if len(payload) < META.size:
        return None
    t = META.unpack_from(payload)
    name = payload[META.size : META.size + t[3]].decode("utf-8", "replace")
    keys = "proto comp flags nlen orig stored sha d total s pct kmax cols rows w h version ecc px quiet crc".split()
    m = dict(zip(keys, t))
    m["name"] = name
    m["ecc"] = ECC_NAMES[m["ecc"]] if m["ecc"] < 4 else "L"
    return m


class Decoder:
    def __init__(self):
        self.det = cv2.QRCodeDetectorAruco()
        self.files = {}
        self.grid = None  # (cols, rows) once metadata is known
        self.stats = dict(frames=0, symbols=0, bad=0, dup=0, good=0, frames_unread=0)

    # ---- QR detection ----------------------------------------------------
    def _single(self, gray):
        """One symbol. Retry failures on a 2x nearest-neighbour upscale: measured 82% -> 100% at v34/3px
        (cubic/linear upscaling blurs the modules and is far worse; the standard detector is not usable)."""
        for scale in (1, 2):
            img = gray if scale == 1 else cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            try:
                b, _, _ = self.det.detectAndDecodeBytes(img)
            except cv2.error:
                continue
            if b:
                return [b]
        return []

    @staticmethod
    def _pad(tile):
        return cv2.copyMakeBorder(tile, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)

    def _quads(self, gray):
        try:
            ok, pts = self.det.detectMulti(gray)
        except cv2.error:
            return []
        return [p.reshape(-1, 2) for p in pts] if ok and pts is not None else []

    def _tiles(self, gray, cols, rows, skip=()):
        """Decode grid cells, skipping any cell that already contains one of the `skip` centre points."""
        h, w = gray.shape
        cw, ch = w / cols, h / rows
        out = []
        for r in range(rows):
            for c in range(cols):
                x0, x1, y0, y1 = c * cw, (c + 1) * cw, r * ch, (r + 1) * ch
                if any(x0 <= cx < x1 and y0 <= cy < y1 for cx, cy in skip):
                    continue
                out += self._single(self._pad(gray[int(y0) : int(y1), int(x0) : int(x1)]))
        return out

    @staticmethod
    def _valid(raw_list):
        n = 0
        for r in raw_list:
            b = _fix_decoded(r)
            n += bool(b) and unpack_packet(whiten(b)) is not None
        return n

    def read_symbols(self, gray):
        """Locate every symbol with detectMulti, decode each from its own crop (no layout assumptions);
        once the grid is known from the metadata, also try cells where no symbol was located."""
        h, w = gray.shape
        raw, centres = [], []
        for quad in self._quads(gray):
            x0, y0 = quad.min(0)
            x1, y1 = quad.max(0)
            centres.append(((x0 + x1) / 2, (y0 + y1) / 2))
            tile = gray[max(0, int(y0) - 2) : min(h, int(y1) + 3), max(0, int(x0) - 2) : min(w, int(x1) + 3)]
            raw += self._single(self._pad(tile))
        if self.grid:
            if self._valid(raw) < self.grid[0] * self.grid[1]:
                raw += self._tiles(gray, *self.grid, skip=centres)
        elif not raw:  # nothing located at all and no layout known: try common grids
            for g in GUESS_GRIDS:
                got = self._tiles(gray, *g)
                if got and self._valid(got) == g[0] * g[1]:
                    raw = got
                    break
        return list(dict.fromkeys(raw))

    # ---- packet handling -------------------------------------------------
    def feed(self, raw_list):
        new = 0
        for raw in raw_list:
            self.stats["symbols"] += 1
            b = _fix_decoded(raw)
            p = unpack_packet(whiten(b)) if b else None
            if p is None:
                self.stats["bad"] += 1
                continue
            kind, fid, pid, total, payload = p
            st = self.files.setdefault(fid, FileState())
            if kind == KIND_META:
                if st.meta is None:
                    m = parse_meta(payload)
                    if m and m["proto"] == PROTO_VERSION:
                        st.meta = m
                        self.grid = (m["cols"], m["rows"])
                        new += 1
                continue
            if pid in st.pk:
                self.stats["dup"] += 1
            else:
                st.pk[pid] = payload
                self.stats["good"] += 1
                new += 1
        return new

    def try_finish(self):
        """-> (name, bytes) for the first file that fully reconstructs and verifies, else None."""
        for fid, st in self.files.items():
            m = st.meta
            if not m:
                continue
            blocks, total = block_layout(m["d"], m["kmax"], m["pct"])
            if len(st.pk) < sum(k for _, k, _, _ in blocks):
                continue
            res = self._reconstruct(st, blocks)
            if res is not None:
                return m["name"], res
        return None

    def _reconstruct(self, st, blocks):
        m = st.meta
        s = m["s"]
        chunks = []
        for off, k, mm, _ in blocks:
            data = {i: st.pk[off + i] for i in range(k) if (off + i) in st.pk}
            par = {j: st.pk[off + k + j] for j in range(mm) if (off + k + j) in st.pk}
            if len(data) + len(par) < k:
                return None
            if len(data) == k:
                chunks += [data[i] for i in range(k)]
            else:
                pad = lambda b: np.frombuffer(b.ljust(s, b"\0"), dtype=np.uint8)
                rows = fec_decode(k, mm, {i: pad(v) for i, v in data.items()},
                                  {j: pad(v) for j, v in par.items()}, s)
                if rows is None:
                    return None
                chunks += [rows[i].tobytes() for i in range(k)]
        stored = b"".join(chunks)[: m["stored"]]
        if len(stored) != m["stored"] or zlib.crc32(stored) & 0xFFFFFFFF != m["crc"]:
            return None
        if m["comp"] == COMP_ZSTD:
            if zstd is None:
                raise SystemExit("stream is zstd-compressed but the 'zstandard' package is not installed")
            try:
                data = zstd.ZstdDecompressor().decompress(stored, max_output_size=m["orig"])
            except zstd.ZstdError:
                return None
        elif m["comp"] == COMP_NONE:
            data = stored
        else:
            raise SystemExit(f"unknown compression id {m['comp']}")
        if len(data) != m["orig"] or hashlib.sha256(data).digest() != m["sha"]:
            return None
        return data


def decode_video(path: str, progress=True):
    """-> (name, data) or (None, None); plus stats dict."""
    cap = _open_capture(path)
    if cap is None:
        raise SystemExit(f"cannot open video: {path}")
    dec = Decoder()
    t0 = time.time()
    result = None
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dec.stats["frames"] += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        n = dec.feed(dec.read_symbols(gray))
        if n == 0:
            dec.stats["frames_unread"] += 1
        if progress:
            print(f"\r  frame {dec.stats['frames']}/{nframes or '?'}  packets {dec.stats['good']}",
                  end="", file=sys.stderr, flush=True)
        if n:
            result = dec.try_finish()
            if result:
                break
    cap.release()
    if progress:
        print(file=sys.stderr)
    dec.stats["time"] = time.time() - t0
    dec.stats["complete"] = result is not None
    if result is None:
        return None, None, dec
    return result[0], result[1], dec


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def throughput(cfg: Cfg):
    """Honest payload numbers: file bytes per second after FEC and repeat-pass overhead."""
    raw_frame = cfg.payload * cfg.cells
    raw = raw_frame * cfg.fps
    eff = raw / (1 + cfg.fec / 100) / cfg.passes
    return raw_frame, raw, eff


def kb(x):
    return f"{x / 1000:,.0f} KB/s"


def print_report(cfg: Cfg, codec: str, extra: dict = None):
    raw_frame, raw, eff = throughput(cfg)
    rows = [
        ("Resolution", f"{cfg.width}x{cfg.height}"),
        ("Grid", f"{cfg.cols}x{cfg.rows}"),
        ("QRs/frame", cfg.cells),
        ("QR version", f"{cfg.version}  ({qr_modules(cfg.version)} modules, {cfg.px}px/module)"),
        ("ECC", cfg.ecc),
        ("Payload/QR", f"{cfg.payload:,} bytes"),
        ("Raw payload/frame", f"{raw_frame:,} bytes"),
        ("FPS", cfg.fps),
        ("Raw throughput", kb(raw)),
        ("FEC overhead", f"{cfg.fec}%"),
        ("Passes", cfg.passes),
        ("Effective throughput", "~" + kb(eff)),
        ("Codec", codec),
    ]
    for k, v in (extra or {}).items():
        rows.append((k, v))
    for k, v in rows:
        print(f"{k + ':':<24}{v}")


def parse_wh(s, sep="x"):
    try:
        a, b = s.lower().split(sep)
        return int(a), int(b)
    except Exception:
        raise SystemExit(f"expected WxH like 1920x1080, got {s!r}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cfg_from_args(a) -> Cfg:
    w, h = parse_wh(a.resolution)
    cols, rows = parse_wh(a.grid)
    return Cfg(width=w, height=h, cols=cols, rows=rows, version=a.version, ecc=a.ecc.upper(), px=a.module_px,
               quiet=a.quiet, fps=a.fps, payload=a.payload, fec=a.fec, passes=a.redundancy,
               meta_every=a.meta_every, codec=a.codec, level=a.level, compress=not a.no_compress,
               workers=a.workers)


def cmd_encode(a):
    cfg = cfg_from_args(a)
    data = open(a.input, "rb").read()
    print(f"input: {a.input} ({len(data):,} bytes)  sha256 {hashlib.sha256(data).hexdigest()}")
    r = encode_bytes(data, os.path.basename(a.input), a.output, cfg)
    cfg, info = r["cfg"], r["info"]
    print(f"compression: {COMP_NAMES[info['comp']]}  {info['orig']:,} -> {info['stored']:,} bytes; "
          f"{info['data_pkts']} data packets, {info['total']} total ({info['blocks']} FEC block(s))")
    print(f"video: {a.output}  {r['frames']} frames = {r['duration']:.1f}s @ {cfg.fps}fps, "
          f"{r['video_bytes'] / 1e6:.1f} MB, encoded in {r['enc_time']:.1f}s")
    print_report(cfg, r["codec"])
    if a.no_verify:
        return 0
    print("verifying (decoding the video we just wrote)...")
    name, out, dec = decode_video(a.output)
    if out == data:
        print(f"VERIFY OK: decoded {len(out):,} bytes, SHA-256 matches ({dec.stats['time']:.1f}s)")
        return 0
    print("VERIFY FAILED: video did not decode back to the original. Try a larger --module-px, a smaller "
          "--version, a lossless --codec, or run `benchmark`.", file=sys.stderr)
    return 2


def cmd_decode(a):
    name, data, dec = decode_video(a.input)
    s = dec.stats
    print(f"frames read {s['frames']}, symbols {s['symbols']}, good packets {s['good']}, "
          f"duplicates {s['dup']}, rejected {s['bad']}, frames with nothing new {s['frames_unread']}, "
          f"{s['time']:.1f}s")
    if data is None:
        for fid, st in dec.files.items():
            m = st.meta
            if m:
                _, total = block_layout(m["d"], m["kmax"], m["pct"])
                print(f"  file {fid.hex()}: {len(st.pk)}/{total} packets, need >= {m['d']} (metadata present)")
            else:
                print(f"  file {fid.hex()}: {len(st.pk)} packets, NO metadata seen")
        print("DECODE FAILED: not enough valid packets; nothing written.", file=sys.stderr)
        return 2
    with open(a.output, "wb") as f:
        f.write(data)
    print(f"original name: {name}")
    print(f"wrote {a.output} ({len(data):,} bytes), SHA-256 verified: {hashlib.sha256(data).hexdigest()}")
    return 0


def cmd_codecs(a):
    w, h = parse_wh(a.resolution)
    for ext in (".avi", ".mkv"):
        print(f"{ext}:")
        res = probe_codecs(w, h, ext, 3)
        for r in res:
            kind = "LOSSLESS" if r["exact"] else f"lossy (max err {r['maxerr']})"
            print(f"  {r['codec']:<5} {kind:<22} {r['bytes'] / 1e6:6.1f} MB for 3 noise frames")
        missing = [c for c in CODEC_CANDIDATES if c not in {r['codec'] for r in res}]
        if missing:
            print(f"  not usable here: {', '.join(missing)}")
    return 0


def sample_stream(data: bytes, n: int, seed: int) -> bytes:
    """Dense, incompressible test bytes: the PDF's compressed stream, topped up with PRNG bytes."""
    base = zstd.ZstdCompressor(level=3).compress(data) if zstd else data
    rng = np.random.default_rng(seed)
    pad = rng.integers(0, 256, max(0, n - len(base)), dtype=np.uint8).tobytes()
    return (base + pad)[:n]


def run_trial(cfg: Cfg, codec: str, sample: bytes, ext=".avi"):
    """Encode `sample` with FEC off (every packet must be read directly) and decode it with the real decoder."""
    cfg = replace(cfg, fec=0, passes=1, compress=False, codec=codec, meta_every=0)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t" + ext)
        t = time.time()
        r = encode_bytes(sample, "bench.bin", path, cfg, quiet_log=True)
        te = time.time() - t
        t = time.time()
        _, out, dec = decode_video(path, progress=False)
        td_ = time.time() - t
        size = os.path.getsize(path)
    got = max((len(f.pk) for f in dec.files.values()), default=0)
    total = r["info"]["total"]
    return dict(ok=out == sample, got=got, total=total, enc=te, dec=td_, frames=r["frames"], bytes=size,
                symbols=dec.stats["symbols"], bad=dec.stats["bad"])


def cmd_benchmark(a):
    data = open(a.input, "rb").read()
    resolutions = [parse_wh(r) for r in a.resolution.split(",")]
    pxs = [int(x) for x in a.module_px.split(",")]
    eccs = [e.upper() for e in a.ecc.split(",")]
    workers = a.workers or default_workers()
    ext = ".avi"

    # candidate generation, ranked by projected raw payload rate (FPS is just a multiplier)
    cands = []
    for (w, h) in resolutions:
        for cols in range(1, 13):
            for rows in range(1, 9):
                if cols * rows > a.max_qrs:
                    continue
                for px in pxs:
                    for ecc in eccs:
                        vmax = max_fit_version(w, h, cols, rows, px, a.quiet)
                        if vmax < a.min_version:
                            continue
                        for v in sorted({vmax} | {x for x in (40, 36, 32, 28, 24, 20, 16, 12) if a.min_version <= x <= vmax}):
                            cfg = Cfg(width=w, height=h, cols=cols, rows=rows, version=v, ecc=ecc, px=px,
                                      quiet=a.quiet, fps=a.fps, fec=a.fec, workers=workers)
                            cfg = resolve(cfg)
                            cands.append(cfg)
    cands.sort(key=lambda c: (-c.payload * c.cells, c.px * -1, c.version))
    print(f"{len(cands)} geometrically valid configurations; testing best-first (max {a.max_trials} trials)")
    codecs_per_res = {}
    best, trials = None, 0
    log = []
    for cfg in cands:
        if trials >= a.max_trials:
            break
        key = (cfg.width, cfg.height, cfg.px)
        if key not in codecs_per_res:
            codecs_per_res[key] = probe_codecs(cfg.width, cfg.height, ext, cfg.px)
        pr = codecs_per_res[key]
        if not pr:
            continue
        codec = a.codec if a.codec != "auto" else pr[0]["codec"]
        ncell = cfg.cells
        n_pk = max(a.min_packets, ncell * a.frames)
        sample = sample_stream(data, cfg.payload * n_pk, seed=trials + 1)
        trials += 1
        t = run_trial(cfg, codec, sample, ext)
        raw_frame, raw, _ = throughput(cfg)
        log.append((cfg, t))
        mark = "PASS" if t["ok"] else "fail"
        print(f"  [{trials:>3}] {cfg.width}x{cfg.height} grid {cfg.cols}x{cfg.rows} v{cfg.version}-{cfg.ecc} "
              f"{cfg.px}px  payload {cfg.payload:>5}  raw {raw / 1000:>6,.0f} KB/s  "
              f"read {t['got']}/{t['total']}  {mark}  ({t['enc']:.1f}s enc, {t['dec']:.1f}s dec)", flush=True)
        if t["ok"]:
            # confirm with a different payload so we do not crown a lucky run
            sample2 = sample_stream(data[::-1], cfg.payload * n_pk, seed=999 + trials)
            t2 = run_trial(cfg, codec, sample2, ext)
            if t2["ok"]:
                best = (cfg, codec, t, t2)
                break
            print("        confirmation run failed, backing off")
    if not best:
        print("No configuration achieved 100% recovery within the trial budget.")
        return 2
    cfg, codec, t, t2 = best
    raw_frame, raw, eff = throughput(replace(cfg, fec=a.fec))
    print()
    print_report(replace(cfg, fec=a.fec), codec, {
        "Decode success": f"100% ({t['got']}/{t['total']} packets read directly, 2 independent runs, FEC not relied on)",
        "Video size": f"{t['bytes'] / t['frames'] / 1000:,.0f} KB/frame",
        "Decode speed": f"{t['frames'] / t['dec']:.1f} frames/s ({t['frames'] * cfg.cells / t['dec']:.0f} QR/s)",
    })
    # codec comparison for the winning geometry
    print("\nCodec comparison on the winning configuration:")
    for r in probe_codecs(cfg.width, cfg.height, ext, cfg.px):
        sample = sample_stream(data, cfg.payload * max(a.min_packets, cfg.cells * 2), seed=5)
        tt = run_trial(cfg, r["codec"], sample, ext)
        print(f"  {r['codec']:<5} {'lossless' if r['exact'] else 'lossy   '} "
              f"{'100%' if tt['ok'] else str(round(100 * tt['got'] / tt['total'])) + '%':>5} recovered, "
              f"{tt['bytes'] / tt['frames'] / 1000:,.0f} KB/frame")
    est_frames = math.ceil(len(data) / max(1, cfg.payload) * (1 + a.fec / 100) / cfg.cells) + 1
    print(f"\nProjected for {a.input} ({len(data):,} bytes, before zstd): ~{est_frames} frames "
          f"= {est_frames / cfg.fps:.1f}s at {cfg.fps}fps")
    print("\nRecommended command:")
    print(f"  python qrstream.py encode {a.input} out.avi --resolution {cfg.width}x{cfg.height} "
          f"--grid {cfg.cols}x{cfg.rows} --version {cfg.version} --module-px {cfg.px} --ecc {cfg.ecc} "
          f"--fps {cfg.fps} --fec {a.fec or 20} --codec {codec}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="PDF <-> tiled QR-code video (OpenCV only, no FFmpeg binary)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="file -> QR video")
    e.add_argument("input")
    e.add_argument("output", help="video file (.avi recommended, .mkv works)")
    e.add_argument("--resolution", default="1920x1080")
    e.add_argument("--grid", default="4x2", help="COLSxROWS of QR symbols per frame")
    e.add_argument("--fps", type=int, default=30)
    e.add_argument("--ecc", default="L", choices=list("LMQHlmqh"))
    e.add_argument("--version", type=int, default=0, help="QR version 1-40 (0 = largest that fits)")
    e.add_argument("--module-px", type=int, default=3, help="video pixels per QR module")
    e.add_argument("--quiet", type=int, default=2, help="quiet-zone modules around each QR")
    e.add_argument("--payload", type=int, default=0, help="file bytes per QR (0 = max)")
    e.add_argument("--fec", type=int, default=20, help="%% Reed-Solomon parity packets (0 = off)")
    e.add_argument("--redundancy", type=int, default=1, help="number of full passes (each in a different order)")
    e.add_argument("--meta-every", type=int, default=8, help="re-send metadata every N frames")
    e.add_argument("--codec", default="auto", help="FOURCC or 'auto' (best lossless that actually opens)")
    e.add_argument("--level", type=int, default=19, help="zstd level")
    e.add_argument("--no-compress", action="store_true")
    e.add_argument("--workers", type=int, default=0, help="QR rendering processes (0 = auto)")
    e.add_argument("--no-verify", action="store_true", help="skip the decode-back check after encoding")
    e.set_defaults(fn=cmd_encode)

    d = sub.add_parser("decode", help="QR video -> file (written only after SHA-256 verifies)")
    d.add_argument("input")
    d.add_argument("output")
    d.set_defaults(fn=cmd_decode)

    b = sub.add_parser("benchmark", help="find the fastest configuration that decodes 100%% on this machine")
    b.add_argument("input")
    b.add_argument("--resolution", default="1920x1080", help="comma list, e.g. 1920x1080,2560x1440,3840x2160")
    b.add_argument("--module-px", default="2,3,4", help="comma list of pixels/module to try")
    b.add_argument("--ecc", default="L", help="comma list, e.g. L,M")
    b.add_argument("--quiet", type=int, default=2)
    b.add_argument("--fps", type=int, default=30)
    b.add_argument("--fec", type=int, default=20, help="FEC %% used for the effective-throughput figure")
    b.add_argument("--frames", type=int, default=3, help="frames per trial")
    b.add_argument("--min-packets", type=int, default=24, help="minimum QR symbols per trial")
    b.add_argument("--max-qrs", type=int, default=48, help="max QR symbols per frame to consider")
    b.add_argument("--min-version", type=int, default=10)
    b.add_argument("--max-trials", type=int, default=40)
    b.add_argument("--codec", default="auto")
    b.add_argument("--workers", type=int, default=0)
    b.set_defaults(fn=cmd_benchmark)

    c = sub.add_parser("codecs", help="list codecs cv2.VideoWriter can really open here")
    c.add_argument("--resolution", default="1920x1080")
    c.set_defaults(fn=cmd_codecs)

    a = ap.parse_args(argv)
    if not hasattr(cv2.QRCodeDetectorAruco, "detectAndDecodeBytes"):
        raise SystemExit("this OpenCV build lacks QRCodeDetectorAruco.detectAndDecodeBytes (needed for raw binary "
                         "payloads); upgrade: pip install -U opencv-python-headless")
    try:
        return a.fn(a)
    finally:
        if _pool:
            _pool.shutdown(cancel_futures=True)


if __name__ == "__main__":
    sys.exit(main())

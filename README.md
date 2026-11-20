# qrstream
A fun experiment testing data transmission over QR
Convert a PDF (or any file) into a video of tiled QR codes, and back again.
Pure Python + OpenCV: no FFmpeg binary, no Base64, works fully offline.

```
file -> SHA-256 -> zstd (only if smaller) -> fixed-size binary chunks
     -> Cauchy Reed-Solomon erasure code over GF(256), across packets
     -> packets (28-byte header, CRC32) -> whitening -> QR (byte mode, raw bytes)
     -> several QR symbols tiled per frame -> cv2.VideoWriter (lossless codec if available)
```

Decoding writes the output file **only after its SHA-256 verifies**.

## Install

```
pip install opencv-contrib-python numpy qrcode zstandard
```

- `zstandard` is optional for encoding, but **required to decode** a stream that was compressed.
- Needs an OpenCV build with `QRCodeDetectorAruco.detectAndDecodeBytes`. If you get an error about it, run `pip install -U opencv-python-headless` (or the contrib build).
- Python 3.9+.

## Usage

```
python qrstream.py encode    input.pdf out.avi --resolution 1920x1080 --grid 4x2 --fec 20
python qrstream.py decode    out.avi recovered.pdf
python qrstream.py benchmark input.pdf
python qrstream.py codecs    [--resolution 1920x1080]
```

`encode` decodes the video it just wrote and checks the SHA-256 (skip with `--no-verify`).
`decode` stops as soon as enough packets are read to rebuild and verify the file.

### Commands

| Command | Purpose |
|---|---|
| `encode` | File to QR video |
| `decode` | QR video to file |
| `benchmark` | Search for the fastest configuration that decodes 100% on this machine, then print a recommended `encode` command |
| `codecs` | List the codecs `cv2.VideoWriter` can actually open here, and whether they are lossless |

```
python qrstream.py encode doc.pdf out.mp4 --module-px 4 --fec 30
python qrstream.py decode out.mp4 recovered.pdf
```

MP4 normally only supports lossy codecs, which blur small QR modules, so the defaults can fail. I haven't run this on MP4. The built-in verify step will tell you straight away: `VERIFY OK` means it's fine, `VERIFY FAILED` means use the safer settings below.

If it fails, use bigger and fewer QRs with more recovery:

```
python qrstream.py encode doc.pdf out.mp4 --module-px 6 --grid 3x2 --version 20 --ecc M --fec 40 --redundancy 2
```

`codecs` and `benchmark` only test `.avi` and `.mkv`, so they won't tell you what works for MP4. If you can use `.avi` or `.mkv`, they're much more reliable and allow the default settings.

### Main `encode` options

| Option | Default | Meaning |
|---|---|---|
| `--resolution` | `1920x1080` | Video size |
| `--grid` | `4x2` | QR symbols per frame (COLSxROWS) |
| `--fps` | `30` | Frame rate |
| `--ecc` | `L` | QR error correction: L, M, Q, H |
| `--version` | `0` | QR version 1-40 (0 = largest that fits) |
| `--module-px` | `3` | Video pixels per QR module |
| `--quiet` | `2` | Quiet-zone modules around each QR |
| `--payload` | `0` | File bytes per QR (0 = max for the version) |
| `--fec` | `20` | Percent Reed-Solomon parity packets (0 = off) |
| `--redundancy` | `1` | Full retransmission passes, each in a different order |
| `--meta-every` | `8` | Re-send metadata every N frames |
| `--codec` | `auto` | FOURCC, or `auto` for the best lossless codec that opens |
| `--level` | `19` | zstd level |
| `--no-compress` | off | Skip zstd |
| `--workers` | `0` | QR rendering processes (0 = auto) |
| `--no-verify` | off | Skip the decode-back check |

## Recommended workflow

1. `python qrstream.py codecs` to see which lossless codecs work on your machine.
2. `python qrstream.py benchmark input.pdf` to find the fastest reliable settings.
3. Run the `encode` command it prints.

## Format details

**Packet** (all big-endian): `magic "QRSV"` (4) | version (1) | kind (1) | file_id (8) | packet_id (4) | total (4) | payload_len (2) | crc32 (4), followed by the payload. The CRC covers the header (with the CRC field zeroed) and the payload. `file_id` is the first 8 bytes of the file's SHA-256.

**Whitening**: each QR payload is XORed with a fixed keystream derived from `sha256(b"qrstream-whiten" + counter)`. This avoids long runs of identical modules, which is why a fixed QR mask pattern (2) is safe and fast.

**Metadata packet** (`kind = 0`): proto, compression, flags, name length, original size, stored size, SHA-256, data packets, total packets, chunk size, FEC percent, k_max, cols, rows, width, height, QR version, ECC, px per module, quiet zone, CRC32 of the stored stream, then the file name. Frame 0 is all metadata and it is refreshed every `--meta-every` frames, in a rotating cell.

**Erasure coding**: systematic Cauchy Reed-Solomon over GF(256) with polynomial `0x11D`, in blocks of at most 255 packets (data + parity). Any `k` of the `k + m` packets in a block rebuild that block.

**Transmission order**: pass 0 interleaves data packets across blocks, then parity. Later passes (`--redundancy`) use deterministic shuffles.

**Decoding**: symbols are located with `detectMulti` and decoded from individual crops, with a 2x nearest-neighbour retry. Once the metadata is read, any cell where no symbol was located is tried on the known grid. Before metadata is known, common grids are guessed. OpenCV returns byte-mode data as Latin-1 text encoded as UTF-8; the decoder reverses that exactly.

## Tips and troubleshooting

- Use a **lossless codec** (FFV1, ULRG, HFYU, LAGS) in `.avi` or `.mkv`. Lossy codecs blur 3px modules; if you must use one, raise `--module-px`, lower `--version`, or use a higher `--ecc`.
- If `VERIFY FAILED` appears: try a larger `--module-px`, a smaller `--version`, a different `--codec`, or run `benchmark`.
- If decoding reports `DECODE FAILED: not enough valid packets`, the summary shows packets read versus needed. Re-encode with more `--fec` or `--redundancy`.
- Heavily re-encoded, rescaled or filmed video is outside what the defaults assume. Lower the grid density (fewer, larger QRs) in that case.
- `benchmark` results are specific to your machine's codecs and OpenCV build.

## Limits

- Output is only written after SHA-256 verification, so a partial video produces no file.
- Throughput depends on resolution, grid, QR version, FPS, FEC and passes. `encode` and `benchmark` print the effective figure.

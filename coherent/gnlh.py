"""Reader for granolasdr --record GNLH captures (complex FT8/JS8 composite).

The file is a 32-byte header (see gm/buffer/BufferFile.h) followed by raw
complex64 blocks of the composite stream (409.6 kHz, full phase). We memmap the
data so multi-hundred-MB recordings cost nothing to open.
"""
import struct
import numpy as np

HEADER_FMT  = "<IIIIQII"   # magic,version,block_samples,element_bytes,interval_ns,sr,rx_sr
HEADER_SIZE = 32
MAGIC       = 0x474E4C48   # 'GNLH'

# FT8/JS8 sub-band layout, mirrors gm/hf/hf_bands.h: (name, wb_start, wb_end).
# Wideband FFT bin = freq_Hz / 100. The composite packs these contiguously, so
# band k occupies composite bins [cum_k, cum_k+bw_k); composite bin = freq/100
# in the recorded stream too (4096-pt composite IFFT, 100 Hz/bin).
HF_BANDS = [
    ("160m",  18390,  18460), ("80m",  35720,  35820), ("60m",  53560,  53740),
    ("40m",   70680,  70820), ("40m-2",71090,  71160), ("30m", 101290, 101420),
    ("20m",  140700, 141000), ("17m", 180700, 181130), ("15m", 210720, 210950),
    ("12m",  249140, 249310), ("10m", 280730, 281000), ("6m",  503110, 503200),
]
# FT8 dial wideband bin per band (freq/100), for locating the FT8 sub-region.
FT8_DIAL = {"160m":18400,"80m":35730,"40m":70740,"30m":101360,"20m":140740,
            "17m":181000,"15m":210740,"12m":249150,"10m":280740,"6m":503130}


def open_recording(path):
    """Return dict with a memmapped complex64 `samples` array + header fields."""
    with open(path, "rb") as f:
        magic, ver, bs, eb, ns, sr, rxsr = struct.unpack(HEADER_FMT, f.read(HEADER_SIZE))
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x} (expected {MAGIC:#x})")
    if eb != 8:
        raise ValueError(f"element_bytes={eb}, expected 8 (complex64)")
    samples = np.memmap(path, dtype=np.complex64, mode="r", offset=HEADER_SIZE)
    return dict(samples=samples, sample_rate=sr, block_samples=bs,
                block_interval_ns=ns, rx_sample_rate=rxsr, version=ver)


def band_layout():
    """List of (name, comp_bin_start, bw, wb_start) with cumulative composite offsets."""
    out, cum = [], 0
    for name, ws, we in HF_BANDS:
        bw = we - ws
        out.append((name, cum, bw, ws))
        cum += bw
    return out


def band_bin_hz(sample_rate, nfft):
    """Hz per composite bin for an nfft-point FFT of the recorded stream."""
    return sample_rate / nfft

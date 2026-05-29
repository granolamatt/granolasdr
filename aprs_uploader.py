#!/usr/bin/env python3
"""
FT8 + JS8 → APRS-IS uploader.

Two functions:
  1. Station spotting: decoded FT8/JS8 callsigns are sent to APRS-IS as
     APRS object packets with a position (sender's grid when available,
     else receiver's grid).
  2. @APRSIS relay: JS8 messages directed to @APRSIS have their payload
     forwarded to APRS-IS as a normal APRS message.

Subscribes to ZMQ PUB sockets (default: 5580 FT8, 5590 JS8).

Usage:
    python3 aprs_uploader.py --call KF0RRR --grid EM73
    python3 aprs_uploader.py --call KF0RRR --grid EM73 --test
    python3 aprs_uploader.py --call KF0RRR --grid EM73 --port 5590  # JS8 only
"""

import argparse
import json
import re
import socket
import time
import zmq

APRS_HOST       = "rotate.aprs.net"
APRS_PORT       = 14580
APRS_TEST_HOST  = "rotate.aprs.net"   # same server, different port
APRS_TEST_PORT  = 23     # echo-only server (read-only for unverified logins)

TOCALL          = "APGRNS"
SOFTWARE_VER    = "granolasdr 1.0"
KEEPALIVE_SEC   = 60
DEDUP_SEC       = 20 * 60   # 20-minute per-callsign spotting dedup
RECONNECT_SEC   = 15


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def aprs_passcode(callsign: str) -> int:
    """Compute APRS-IS passcode (XOR-fold, & 0x7FFF)."""
    call = callsign.upper().split("-")[0]
    n = 0x73E2
    for i, c in enumerate(call):
        n ^= ord(c) << (8 if i % 2 == 0 else 0)
    return n & 0x7FFF


def grid_to_latlon(grid: str) -> tuple[float, float]:
    """Convert 4-char Maidenhead grid to (lat, lon) centre of square."""
    g = grid.upper()
    lon = (ord(g[0]) - ord("A")) * 20 - 180 + (ord(g[2]) - ord("0")) * 2 + 1.0
    lat = (ord(g[1]) - ord("A")) * 10 - 90  + (ord(g[3]) - ord("0"))     + 0.5
    return lat, lon


def latlon_to_aprs(lat: float, lon: float) -> str:
    """Format lat/lon as APRS position string DDmm.mmN/DDDmm.mmW."""
    lat_d = int(abs(lat))
    lat_m = (abs(lat) - lat_d) * 60
    lon_d = int(abs(lon))
    lon_m = (abs(lon) - lon_d) * 60
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{lat_d:02d}{lat_m:05.2f}{ns}/{lon_d:03d}{lon_m:05.2f}{ew}"


_GRID_RE = re.compile(r"\b([A-R]{2}[0-9]{2})\b")
_APRSIS_RE = re.compile(r"@APRSIS\s+(.+)", re.IGNORECASE)


def extract_grid(text: str) -> str | None:
    """Pull first 4-char Maidenhead grid from a decoded message, if present."""
    m = _GRID_RE.search(text)
    return m.group(1) if m else None


def extract_aprsis_payload(text: str) -> str | None:
    """Return the payload of an @APRSIS-directed JS8 message, or None."""
    m = _APRSIS_RE.search(text)
    return m.group(1).strip() if m else None


def aprs_object_packet(
    rx_call: str,
    spotted_call: str,
    lat: float,
    lon: float,
    mode: str,
    freq_hz: float,
    snr: float,
    grid: str,
    unix_ts: float,
) -> str:
    """Build a complete APRS object packet for a spotted station."""
    t = time.gmtime(unix_ts)
    timestamp = f"{t.tm_mday:02d}{t.tm_hour:02d}{t.tm_min:02d}z"
    pos = latlon_to_aprs(lat, lon)
    name = f"{spotted_call:<9s}"[:9]   # pad/trim to exactly 9 chars
    freq_khz = freq_hz / 1000.0
    comment = f"{mode} {freq_khz:.1f}kHz SNR={snr:+.0f}dB"
    if grid:
        comment += f" [{grid}]"
    # Symbol: \ (alternate table) + _ = weather station (neutral spot marker)
    # Using /r = recreational vehicle to distinguish RF spots; common choice is /- (house)
    # We'll use the "diamond" symbol (/) to represent a generic radio station
    pkt = f"{rx_call}>{TOCALL},TCPIP*:;{name}*{timestamp}{pos}\\{comment}"
    return pkt


def aprs_message_packet(
    rx_call: str,
    addressee: str,
    message: str,
    msg_id: int,
) -> str:
    """Build an APRS :message: packet."""
    addr = f"{addressee:<9s}"[:9]
    return f"{rx_call}>{TOCALL},TCPIP*:::{addr}:{message}{{{msg_id}}}"


# ---------------------------------------------------------------------------
# APRS-IS TCP connection
# ---------------------------------------------------------------------------

class AprsConnection:
    def __init__(self, host: str, port: int, login_line: str):
        self._host = host
        self._port = port
        self._login = login_line
        self._sock: socket.socket | None = None
        self._last_send = 0.0

    def _connect(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(10)
                s.connect((self._host, self._port))
                # Read server banner
                banner = s.recv(512).decode("ascii", errors="replace").strip()
                print(f"[aprs] connected to {self._host}:{self._port}: {banner}")
                # Send login
                s.sendall((self._login + "\r\n").encode("ascii"))
                time.sleep(0.5)
                resp = s.recv(512).decode("ascii", errors="replace").strip()
                print(f"[aprs] login resp: {resp}")
                if "unverified" in resp.lower():
                    print("[aprs] WARNING: login unverified — check passcode")
                s.settimeout(5)
                self._sock = s
                self._last_send = time.time()
                return
            except OSError as e:
                print(f"[aprs] connect failed ({e}), retry in {RECONNECT_SEC}s")
                time.sleep(RECONNECT_SEC)

    def ensure_connected(self):
        if self._sock is None:
            self._connect()

    def send(self, line: str):
        """Send one APRS line (adds \\r\\n). Reconnects on failure."""
        self.ensure_connected()
        try:
            self._sock.sendall((line + "\r\n").encode("ascii"))
            self._last_send = time.time()
            # Drain any server responses to avoid buffer filling
            try:
                self._sock.setblocking(False)
                _ = self._sock.recv(4096)
                self._sock.setblocking(True)
                self._sock.settimeout(5)
            except BlockingIOError:
                self._sock.setblocking(True)
                self._sock.settimeout(5)
        except OSError as e:
            print(f"[aprs] send failed ({e}), reconnecting")
            self._sock = None
            self._connect()
            self._sock.sendall((line + "\r\n").encode("ascii"))
            self._last_send = time.time()

    def keepalive(self):
        """Send a comment packet if idle for KEEPALIVE_SEC."""
        if time.time() - self._last_send >= KEEPALIVE_SEC:
            self.send("#keepalive")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", required=True, help="Your callsign (receiver)")
    ap.add_argument("--grid", required=True, help="Your Maidenhead grid (e.g. EM73)")
    ap.add_argument("--pass", dest="passcode", type=int, default=None,
                    help="APRS-IS passcode (auto-computed if omitted)")
    ap.add_argument("--port", type=int, action="append", metavar="PORT",
                    help="ZMQ port (default: 5580 5590; may be repeated)")
    ap.add_argument("--test", action="store_true",
                    help="Print packets to stdout instead of sending to APRS-IS")
    args = ap.parse_args()

    rx_call  = args.call.upper()
    rx_grid  = args.grid.upper()
    passcode = args.passcode if args.passcode is not None else aprs_passcode(rx_call)
    zmq_ports = args.port if args.port else [5580, 5590]

    rx_lat, rx_lon = grid_to_latlon(rx_grid)

    print(f"[aprs] receiver {rx_call} / {rx_grid}  passcode={passcode}")
    print(f"[aprs] ZMQ ports: {zmq_ports}")
    if args.test:
        print("[aprs] TEST MODE — packets printed to stdout only")

    login_line = (
        f"user {rx_call} pass {passcode} vers {SOFTWARE_VER} "
        f"filter r/{rx_lat:.2f}/{rx_lon:.2f}/2000"
    )

    conn: AprsConnection | None = None
    if not args.test:
        conn = AprsConnection(APRS_HOST, APRS_PORT, login_line)
        conn.ensure_connected()

    ctx = zmq.Context()
    poller = zmq.Poller()
    socks = []
    for port in zmq_ports:
        s = ctx.socket(zmq.SUB)
        s.connect(f"tcp://localhost:{port}")
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        poller.register(s, zmq.POLLIN)
        socks.append(s)

    # dedup: (callsign, band_mhz) → last_spotted_unix
    spotted: dict[tuple, float] = {}
    msg_id = 1

    while True:
        ready = dict(poller.poll(timeout=1000))

        for s in socks:
            if ready.get(s) != zmq.POLLIN:
                continue
            try:
                raw = s.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            call  = msg.get("call", "").strip()
            text  = msg.get("text", "").strip()
            freq  = float(msg.get("freq", 0))
            snr   = float(msg.get("snr", 0))
            unix  = float(msg.get("unix", time.time()))
            mode  = msg.get("mode", "FT8")

            # ---- @APRSIS relay (JS8 only) --------------------------------
            payload = extract_aprsis_payload(text)
            if payload and mode == "JS8":
                pkt = aprs_message_packet(rx_call, "APRS    ", payload, msg_id)
                msg_id += 1
                print(f"[aprs] RELAY  {pkt}")
                if conn:
                    conn.send(pkt)

            # ---- Station spotting ----------------------------------------
            if not call or call.startswith("Error"):
                continue

            band = int(freq) // 1_000_000
            key  = (call.upper(), band)
            now  = time.time()
            if now - spotted.get(key, 0) < DEDUP_SEC:
                continue
            spotted[key] = now

            # Use sender's grid from message text when available (JS8 heartbeats)
            sender_grid = extract_grid(text) if mode == "JS8" else None
            if sender_grid:
                try:
                    lat, lon = grid_to_latlon(sender_grid)
                except (IndexError, ValueError):
                    lat, lon = rx_lat, rx_lon
                    sender_grid = None
            else:
                lat, lon = rx_lat, rx_lon

            pkt = aprs_object_packet(rx_call, call, lat, lon, mode, freq, snr,
                                     sender_grid or rx_grid, unix)
            print(f"[aprs] SPOT   {pkt}")
            if conn:
                conn.send(pkt)

        # Keepalive
        if conn:
            conn.keepalive()


if __name__ == "__main__":
    main()

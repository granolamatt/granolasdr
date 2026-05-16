#!/usr/bin/env python3
"""
FT8 → PSKReporter uploader.
Subscribes to ZMQ PUB socket on tcp://localhost:5580, buffers decoded messages,
and uploads to PSKReporter every UPLOAD_INTERVAL_SEC seconds.

Packet format: https://pskreporter.info/pskdev.html

Usage:
    python3 psk_uploader.py --call W1AW --grid DM78
    python3 psk_uploader.py --call W1AW --grid DM78 --test   # verify at pskreporter.info/analyze.html
"""

import argparse
import json
import socket
import struct
import time
import random
import zmq

UPLOAD_INTERVAL_SEC = 300   # PSKReporter asks no more than once per 5 min
PSK_HOST  = "report.pskreporter.info"
PSK_PORT  = 4739
TEST_PORT = 14739   # packet analyzer — results at pskreporter.info/analyze.html

SOFTWARE  = "granolasdr v1"
DEFAULT_RIG = "rx888"

# ---- Hardcoded template bytes from pskreporter.info/pskdev.html ----------- #
# Receiver template: receiverCallsign, receiverLocator, decodingSoftware, antennaInformation
_RECV_TMPL = bytes.fromhex(
    "0003002C"          # set-id=3, length=44
    "9992000400 01"     # template-id=0x9992, field-count=4, scope-count=1
    "8002FFFF0000768F"  # receiverCallsign    (variable)
    "8004FFFF0000768F"  # receiverLocator     (variable)
    "8008FFFF0000768F"  # decodingSoftware    (variable)
    "8009FFFF0000768F"  # antennaInformation  (variable)
    "0000"              # pad to 4-byte boundary
    .replace(" ", "")
)

# Sender template: senderCallsign, frequency(4), sNR(1), iMD(1), mode, informationSource(1), flowStartSeconds(4)
_SEND_TMPL = bytes.fromhex(
    "0002003C"          # set-id=2, length=60
    "99930007"          # template-id=0x9993, field-count=7
    "8001FFFF0000768F"  # senderCallsign    (variable)
    "8005000400 00768F" # frequency         (4 bytes)
    "8006000100 00768F" # sNR               (1 byte signed)
    "8007000100 00768F" # iMD               (1 byte signed, set to 0)
    "800AFFFF0000768F"  # mode              (variable)
    "800B000100 00768F" # informationSource (1 byte)
    "00960004"          # flowStartSeconds  (4 bytes, standard IPFIX field 150)
    .replace(" ", "")
)

TEMPLATES = _RECV_TMPL + _SEND_TMPL


def _str(s: str) -> bytes:
    """Encode string as 1-byte length prefix + ASCII bytes (max 254 chars)."""
    b = s.encode("ascii")[:254]
    return struct.pack("B", len(b)) + b


def _pad4(data: bytes) -> bytes:
    r = len(data) % 4
    return data + b"\x00" * ((4 - r) % 4)


def _receiver_block(rx_call: str, rx_grid: str, rig: str) -> bytes:
    """99 92 ll ll + {receiverCallsign, receiverLocator, decodingSoftware, antennaInformation}"""
    payload = _pad4(_str(rx_call) + _str(rx_grid) + _str(SOFTWARE) + _str(rig))
    return struct.pack(">HH", 0x9992, 4 + len(payload)) + payload


def _sender_block(reports: list[dict]) -> bytes:
    """99 93 ll ll + one record per report."""
    records = b""
    for r in reports:
        snr = max(-128, min(127, int(round(r["snr"]))))
        records += (
            _str(r["call"]) +
            struct.pack(">I", int(r["freq"])) +
            struct.pack(">b", snr) +
            struct.pack(">b", 0) +          # iMD — not measured, send 0
            _str("FT8") +
            struct.pack(">B", 1) +          # informationSource=1 (auto extracted)
            struct.pack(">I", int(r["unix"]))
        )
    records = _pad4(records)
    return struct.pack(">HH", 0x9993, 4 + len(records)) + records


def build_packet(rx_call: str, rx_grid: str, rig: str, reports: list[dict],
                 seq: int, session_id: int, include_templates: bool) -> bytes:
    body = b""
    if include_templates:
        body += TEMPLATES
    body += _receiver_block(rx_call, rx_grid, rig)
    body += _sender_block(reports)
    # Header: version=0x000A, total_length, unix_time, seq, session_id
    hdr = struct.pack(">HHIII", 0x000A, 16 + len(body), int(time.time()), seq, session_id)
    return hdr + body


def upload(rx_call, rx_grid, rig, reports, seq, session_id, include_templates, host, port):
    pkt = build_packet(rx_call, rx_grid, rig, reports, seq, session_id, include_templates)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(pkt, (host, port))
    print(f"[psk] {len(reports)} reports → {host}:{port}  "
          f"seq={seq}  {len(pkt)}B  templates={'yes' if include_templates else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", required=True, help="Your callsign (receiver)")
    ap.add_argument("--grid", required=True, help="Your Maidenhead grid (e.g. DM78)")
    ap.add_argument("--port", type=int, default=5580, help="ZMQ port (default 5580)")
    ap.add_argument("--rig", default=DEFAULT_RIG, help=f"Receiver hardware (default: {DEFAULT_RIG})")
    ap.add_argument("--test", action="store_true",
                    help="Send to port 14739 (packet analyzer) instead of 4739")
    ap.add_argument("--send-test-packet", action="store_true",
                    help="Send one dummy packet immediately to port 14739 and exit")
    args = ap.parse_args()

    rx_call   = args.call.upper()
    rx_grid   = args.grid.upper()
    rx_rig    = args.rig
    psk_port  = TEST_PORT if args.test else PSK_PORT
    session_id = random.randint(1, 0xFFFFFFFF)
    seq = 0
    packets_sent = 0
    last_template_time = 0.0

    if args.test or args.send_test_packet:
        print("[psk] TEST MODE — check results at https://pskreporter.info/analyze.html")

    if args.send_test_packet:
        dummy = [{"call": rx_call, "freq": 14074000, "snr": -10, "unix": int(time.time())}]
        upload(rx_call, rx_grid, rx_rig, dummy, seq=0, session_id=session_id,
               include_templates=True, host=PSK_HOST, port=TEST_PORT)
        print("[psk] Dummy packet sent. Check https://pskreporter.info/analyze.html")
        return

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://localhost:{args.port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 1000)

    print(f"[psk] receiver {rx_call} / {rx_grid}  rig: {rx_rig}")
    print(f"[psk] session_id=0x{session_id:08X}")
    print(f"[psk] ZMQ tcp://localhost:{args.port} → {PSK_HOST}:{psk_port}")
    print(f"[psk] uploading every {UPLOAD_INTERVAL_SEC}s")

    pending: list[dict] = []
    last_upload = time.time()

    while True:
        try:
            raw = sock.recv_string()
            msg = json.loads(raw)
            if not msg.get("call") or msg["call"].startswith("Error"):
                continue
            pending.append(msg)
        except zmq.Again:
            pass

        if time.time() - last_upload >= UPLOAD_INTERVAL_SEC and pending:
            # Send templates in first 3 packets and then once per hour
            include_tmpl = packets_sent < 3 or time.time() - last_template_time >= 3600
            if include_tmpl:
                last_template_time = time.time()

            upload(rx_call, rx_grid, rx_rig, pending, seq, session_id, include_tmpl,
                   PSK_HOST, psk_port)
            seq += len(pending)   # seq = cumulative report count, not packet count
            packets_sent += 1
            pending.clear()
            last_upload = time.time()


if __name__ == "__main__":
    main()

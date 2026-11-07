#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${HOME}/.venv_puzzle71"
PY_SCRIPT="${VENV_DIR}/puzzle71_search.py"

echo "=== Bitcoin Puzzle #71 Searcher ==="

# 1. Create venv if it doesn't exist
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install -q --disable-pip-version-check coincurve base58 pycryptodome
fi

# 2. Write the Python searcher
cat > "${PY_SCRIPT}" << 'EOF'
#!/usr/bin/env python3
import hashlib
import multiprocessing as mp
import os
import secrets
import signal
import sys
import time

RANGE_PARAM = 25.00
WINDOW_START_PCT = 0.00
WORKERS = os.cpu_count() or 1
CHUNK = 1 << 16

TARGET_ADDRESS = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
TARGET_HASH160 = bytes.fromhex("f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8")
RANGE_START = 0x400000000000000000
RANGE_END = 0x7FFFFFFFFFFFFFFFFF

def get_ripemd():
    try:
        hashlib.new("ripemd160", b"")
        return lambda b: hashlib.new("ripemd160", b).digest()
    except Exception:
        from Crypto.Hash import RIPEMD160
        return lambda b: RIPEMD160.new(b).digest()

def privkey_to_address(privkey_hex):
    import base58
    from coincurve import PrivateKey
    rip = get_ripemd()
    privkey_hex = privkey_hex.strip().lower().replace("0x", "")
    pub = PrivateKey(bytes.fromhex(privkey_hex.zfill(64))).public_key.format(compressed=True)
    versioned = b"\x00" + rip(hashlib.sha256(pub).digest())
    checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
    return base58.b58encode(versioned + checksum).decode()

def compute_window():
    if not (0 < RANGE_PARAM <= 100):
        sys.exit("RANGE_PARAM must be in (0, 100]")
    if WINDOW_START_PCT < 0 or WINDOW_START_PCT + RANGE_PARAM > 100:
        sys.exit("WINDOW_START_PCT + RANGE_PARAM must be <= 100")
    total = RANGE_END - RANGE_START + 1
    scale = 10_000_000
    lo = RANGE_START + total * int(round(WINDOW_START_PCT * 100000)) // scale
    hi = RANGE_START + total * int(round((WINDOW_START_PCT + RANGE_PARAM) * 100000)) // scale - 1
    return lo, min(hi, RANGE_END)

def worker(idx, lo, hi, counts, found_evt, result_q):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {idx % (os.cpu_count() or 1)})
        except Exception:
            pass
    from coincurve import PublicKey
    rip = get_ripemd()
    sha = hashlib.sha256
    combine = PublicKey.combine_keys
    from_secret = PublicKey.from_secret
    target = TARGET_HASH160
    G = from_secret((1).to_bytes(32, "big"))
    span = hi - lo + 1
    chunk = min(CHUNK, span)
    randbelow = secrets.randbelow
    while not found_evt.is_set():
        start = lo + randbelow(span - chunk + 1)
        pub = from_secret(start.to_bytes(32, "big"))
        for i in range(chunk):
            if rip(sha(pub.format()).digest()) == target:
                result_q.put(start + i)
                found_evt.set()
                return
            pub = combine((pub, G))
        counts[idx] += chunk

def main():
    assert privkey_to_address("1") == "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"
    lo, hi = compute_window()
    print(f"Target  : {TARGET_ADDRESS}")
    print(f"Window  : {lo:x} - {hi:x} ({RANGE_PARAM}% from {WINDOW_START_PCT}%)")
    print(f"Workers : {WORKERS}  chunk={CHUNK}  (Ctrl+C to stop)\n")

    counts = mp.Array("Q", WORKERS, lock=False)
    found_evt = mp.Event()
    result_q = mp.Queue()
    procs = [mp.Process(target=worker, args=(i, lo, hi, counts, found_evt, result_q), daemon=True)
             for i in range(WORKERS)]
    for p in procs:
        p.start()

    t0 = time.time()
    try:
        while not found_evt.is_set():
            time.sleep(2)
            n = sum(counts)
            el = time.time() - t0
            print(f"\rTried {n:,} | {n / max(el, 1e-9):,.0f} keys/s | {el:,.0f}s", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\nStopped (Ctrl+C)")
        found_evt.set()

    for p in procs:
        p.join(timeout=3)

    if not result_q.empty():
        k = result_q.get()
        keyhex = f"{k:064x}"
        print(f"\n\nFOUND\nPrivate key: {keyhex}\nAddress    : {privkey_to_address(keyhex)}")
        with open("FOUND.txt", "a") as f:
            f.write(f"{TARGET_ADDRESS} {keyhex}\n")
    else:
        print(f"\nTotal tried: {sum(counts):,}")

if __name__ == "__main__":
    main()
EOF

# 3. Run it
echo "Starting search..."
"${VENV_DIR}/bin/python" "${PY_SCRIPT}"

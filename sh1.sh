#!/usr/bin/env bash
set -euo pipefail

PROJECT=keysearch
mkdir -p "$PROJECT/src" "$PROJECT/.cargo"
cd "$PROJECT"

cat > Cargo.toml << 'EOF'
[package]
name = "keysearch"
version = "0.1.0"
edition = "2021"

[dependencies]
rand = "0.8"
ripemd = "0.1"
secp256k1 = "0.29"
sha2 = { version = "0.10", features = ["asm"] }
bs58 = { version = "0.5", features = ["check"] }
core_affinity = "0.8"
ctrlc = "3"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
debug = false
EOF

cat > .cargo/config.toml << 'EOF'
[build]
rustflags = ["-C", "target-cpu=native"]
EOF

cat > src/main.rs << 'EOF'
use rand::Rng;
use ripemd::Ripemd160;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const RANGE_PARAM: f64 = 25.00;
const WINDOW_START_PCT: f64 = 0.00;
const CHUNK: u128 = 1 << 20;
const BATCH: u64 = 1 << 12;

const TARGET_ADDRESS: &str = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU";
const TARGET_HASH160: [u8; 20] = [
    0xf6, 0xf5, 0x43, 0x1d, 0x25, 0xbb, 0xf7, 0xb1, 0x2e, 0x8a, 0xdd, 0x9a, 0xf5, 0xe3, 0x47,
    0x5c, 0x44, 0xa0, 0xa5, 0xb8,
];
const RANGE_START: u128 = 0x40_0000_0000_0000_0000;
const RANGE_END: u128 = 0x7f_ffff_ffff_ffff_ffff;

#[repr(align(64))]
struct Counter(AtomicU64);

#[inline(always)]
fn hash160(pubkey: &[u8; 33]) -> [u8; 20] {
    let s = Sha256::digest(pubkey);
    let r = Ripemd160::digest(s);
    r.into()
}

#[inline(always)]
fn key_bytes(k: u128) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[16..].copy_from_slice(&k.to_be_bytes());
    b
}

fn privkey_to_address(k: u128) -> String {
    let secp = Secp256k1::signing_only();
    let sk = SecretKey::from_slice(&key_bytes(k)).unwrap();
    let pk = PublicKey::from_secret_key(&secp, &sk).serialize();
    let h = hash160(&pk);
    let mut v = Vec::with_capacity(21);
    v.push(0u8);
    v.extend_from_slice(&h);
    bs58::encode(v).with_check().into_string()
}

fn compute_window(pct: f64, start_pct: f64) -> (u128, u128) {
    if !(pct > 0.0 && pct <= 100.0) {
        eprintln!("range percent must be in (0, 100]");
        std::process::exit(1);
    }
    if start_pct < 0.0 || start_pct + pct > 100.0 + 1e-9 {
        eprintln!("window start + range percent must be <= 100");
        std::process::exit(1);
    }
    let total = RANGE_END - RANGE_START + 1;
    let scale: u128 = 10_000_000;
    let a = (start_pct * 100_000.0).round() as u128;
    let b = ((start_pct + pct) * 100_000.0).round() as u128;
    let lo = RANGE_START + total * a / scale;
    let hi = (RANGE_START + total * b / scale - 1).min(RANGE_END);
    (lo, hi)
}

fn worker(
    idx: usize,
    lo: u128,
    hi: u128,
    counters: Arc<Vec<Counter>>,
    stop: Arc<AtomicBool>,
    found: Arc<Mutex<Option<u128>>>,
) {
    if let Some(ids) = core_affinity::get_core_ids() {
        if !ids.is_empty() {
            core_affinity::set_for_current(ids[idx % ids.len()]);
        }
    }
    let secp = Secp256k1::signing_only();
    let g = PublicKey::from_secret_key(&secp, &SecretKey::from_slice(&key_bytes(1)).unwrap());
    let span = hi - lo + 1;
    let chunk = (CHUNK.min(span)) as u64;
    let mut rng = rand::thread_rng();
    let counter = &counters[idx].0;

    while !stop.load(Ordering::Relaxed) {
        let start = lo + rng.gen_range(0..=(span - chunk as u128));
        let sk = SecretKey::from_slice(&key_bytes(start)).unwrap();
        let mut pk = PublicKey::from_secret_key(&secp, &sk);

        let mut done: u64 = 0;
        while done < chunk {
            let n = BATCH.min(chunk - done);
            for j in 0..n {
                if hash160(&pk.serialize()) == TARGET_HASH160 {
                    *found.lock().unwrap() = Some(start + (done + j) as u128);
                    stop.store(true, Ordering::SeqCst);
                    return;
                }
                pk = match pk.combine(&g) {
                    Ok(p) => p,
                    Err(_) => {
                        counter.fetch_add(j, Ordering::Relaxed);
                        done = chunk;
                        break;
                    }
                };
            }
            if done < chunk {
                counter.fetch_add(n, Ordering::Relaxed);
                done += n;
            }
            if stop.load(Ordering::Relaxed) {
                return;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pct: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(RANGE_PARAM);
    let start_pct: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(WINDOW_START_PCT);
    let workers: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    if privkey_to_address(1) != "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH" {
        eprintln!("self-test failed");
        std::process::exit(1);
    }

    let (lo, hi) = compute_window(pct, start_pct);
    println!("Target  : {}", TARGET_ADDRESS);
    println!("Window  : {:x} - {:x} ({}% from {}%)", lo, hi, pct, start_pct);
    println!("Workers : {}  chunk={}  (Ctrl+C to stop)\n", workers, CHUNK);

    let stop = Arc::new(AtomicBool::new(false));
    let found = Arc::new(Mutex::new(None));
    let counters: Arc<Vec<Counter>> =
        Arc::new((0..workers).map(|_| Counter(AtomicU64::new(0))).collect());

    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst)).unwrap();
    }

    let handles: Vec<_> = (0..workers)
        .map(|i| {
            let (c, s, f) = (counters.clone(), stop.clone(), found.clone());
            thread::spawn(move || worker(i, lo, hi, c, s, f))
        })
        .collect();

    let t0 = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(500));
        let n: u64 = counters.iter().map(|c| c.0.load(Ordering::Relaxed)).sum();
        let el = t0.elapsed().as_secs_f64().max(1e-9);
        print!("\rTried {} | {:.0} keys/s | {:.0}s   ", n, n as f64 / el, el);
        std::io::stdout().flush().ok();
    }

    for h in handles {
        h.join().ok();
    }

    let total: u64 = counters.iter().map(|c| c.0.load(Ordering::Relaxed)).sum();
    let result = found.lock().unwrap().clone();
    match result {
        Some(k) => {
            let line = format!("{} {:064x}", TARGET_ADDRESS, k);
            println!("\n\nFOUND\nPrivate key: {:064x}\nAddress    : {}", k, privkey_to_address(k));
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("FOUND.txt") {
                writeln!(f, "{}", line).ok();
            }
        }
        None => println!("\n\nStopped. Total tried: {}", total),
    }
}
EOF

echo "Building release binary (this may take a minute)..."
cargo build --release

echo "Starting keysearch..."
exec ./target/release/keysearch "$@"

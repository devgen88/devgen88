#!/usr/bin/env bash
set -euo pipefail

# keysearch — Bitcoin puzzle #71 lottery search, CPU-optimized build.
#
# vs the previous version, per-key cost drops ~30x on a modern x86-64 core:
#   * custom secp256k1 field arithmetic (4x u64 limbs, fast p = 2^256-2^32-977
#     reduction), fully unrolled schoolbook multiply
#   * affine point addition with Montgomery BATCH inversion: one inversion
#     per 1024 keys instead of one per key (~6 field muls per key total,
#     vs a full Jacobian add + inversion per key before)
#   * 1024 independent "strands" per chunk make the batch inversion possible
#     while still covering contiguous keys (chunk = 2^20, same as before)
#   * single-block SHA-256 (33-byte input) with SHA-NI when available
#   * single-block RIPEMD-160 (32-byte input), 8-way AVX2 when available
#   * every hand-rolled primitive is verified against the sha2/ripemd/
#     secp256k1 crates at startup and once per chunk; every hit is fully
#     re-derived with the crates before being recorded
#
# CLI (unchanged): keysearch [range_percent] [start_percent] [workers]
# Env: KEYSEARCH_TEST=1    run self-tests and exit
#      KEYSEARCH_SCALAR=1  force scalar hash implementations

# Install Rust only if missing
if ! command -v cargo >/dev/null 2>&1; then
  echo "Installing Rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

PROJECT=keysearch
mkdir -p "$PROJECT/src" "$PROJECT/.cargo"
cd "$PROJECT"

cat > Cargo.toml << 'TOML_EOF'
[package]
name = "keysearch"
version = "0.1.0"
edition = "2021"

[dependencies]
rand = "0.8"
ripemd = "0.1"
secp256k1 = "0.29"
sha2 = "0.10"
bs58 = { version = "0.5", features = ["check"] }
core_affinity = "0.8"
ctrlc = "3"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
debug = false
TOML_EOF

cat > .cargo/config.toml << 'CFG_EOF'
# NOTE: deliberately NOT using target-cpu=native. All fast paths (SHA-NI,
# AVX2) are explicit intrinsics with runtime CPU detection; letting the
# compiler auto-vectorize the scalar code with native's full ISA (e.g.
# AVX-512 on some Xeons/VMs) measured 15-60x SLOWER when mixed with the
# legacy-SSE SHA-NI calls on some machines. A generic baseline plus explicit
# intrinsics is both faster and more predictable here.
[build]
rustflags = []
CFG_EOF

cat > src/main.rs << 'RUST_EOF'
// keysearch — Bitcoin puzzle #71 lottery search, CPU-optimized.
//
// Same CLI and behavior as the original, but with a much lower cost per key:
//
//   * custom secp256k1 field arithmetic: 4x u64 limbs with fast reduction for
//     p = 2^256 - 2^32 - 977 (no bignum crate, no C call, no allocation)
//   * affine point addition with Montgomery *batch inversion*: ~6 field muls
//     per key instead of a full Jacobian add + one modular inversion per key
//     (an inversion costs ~380 muls — the old code paid that on every key)
//   * 1024 independent "strands" per chunk make batch inversion possible while
//     still covering a contiguous run of 2^20 keys per chunk (strand j of
//     round t holds key k0 + j + t*1024; all strands step by the same
//     precomputed point 1024*G, so 1024 additions per round share ONE inversion)
//   * specialized single-block SHA-256 (fixed 33-byte input, padding folded
//     into constants) with a SHA-NI path when the CPU has it
//   * specialized single-block RIPEMD-160 (fixed 32-byte input) with an
//     8-way AVX2 path when the CPU has it
//   * fail-safe: every hand-rolled primitive is cross-checked against the
//     sha2 / ripemd / secp256k1 crates at startup, the hash pipeline is
//     re-verified against the crates once per chunk, and every candidate hit
//     is fully re-derived with the crates before being recorded. A SIMD path
//     that fails its self-test is disabled (falls back to scalar), never
//     trusted.
//
// CLI (unchanged): keysearch [range_percent] [start_percent] [workers]
// Env: KEYSEARCH_TEST=1  run self-tests and exit
//      KEYSEARCH_SCALAR=1  force scalar hash implementations

use rand::Rng;
use ripemd::Ripemd160;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::Write;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// FIX #1: default is the FULL range, not the lowest 25%.
const RANGE_PARAM: f64 = 100.0;
const WINDOW_START_PCT: f64 = 0.0;

const STRANDS: usize = 1024; // batch-inversion width (also the step between lanes)
const ROUNDS: u64 = (1 << 20) / STRANDS as u64; // 1024 rounds -> 2^20 keys per chunk
const SCALE: u128 = 10_000_000; // 100% * 1e5 resolution

const TARGET_ADDRESS: &str = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU";
const TARGET_HASH160: [u8; 20] = [
    0xf6, 0xf5, 0x43, 0x1d, 0x25, 0xbb, 0xf7, 0xb1, 0x2e, 0x8a, 0xdd, 0x9a, 0xf5, 0xe3, 0x47,
    0x5c, 0x44, 0xa0, 0xa5, 0xb8,
];
const RANGE_START: u128 = 0x40_0000_0000_0000_0000;
const RANGE_END: u128 = 0x7f_ffff_ffff_ffff_ffff;

#[repr(align(64))]
struct Counter(AtomicU64);

fn die(msg: &str) -> ! {
    eprintln!("{}", msg);
    std::process::exit(1);
}

// ============================= field arithmetic =============================
// secp256k1 base field: p = 2^256 - 2^32 - 977, values kept in [0, p).
// 4 little-endian u64 limbs. 2^256 = C (mod p) with C = 2^32 + 977.

mod fe {
    pub const P: [u64; 4] = [0xFFFF_FFFE_FFFF_FC2F, !0, !0, !0];
    pub const C: u128 = 0x1_0000_03D1; // 2^256 mod p

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct Fe(pub [u64; 4]);

    impl Fe {
        pub const ZERO: Fe = Fe([0; 4]);
        pub const ONE: Fe = Fe([1, 0, 0, 0]);

        #[inline(always)]
        pub fn is_zero(&self) -> bool {
            self.0 == [0; 4]
        }

        pub fn from_be_bytes(b: &[u8; 32]) -> Fe {
            let mut l = [0u64; 4];
            for i in 0..4 {
                l[i] = u64::from_be_bytes(b[8 * (3 - i)..8 * (3 - i) + 8].try_into().unwrap());
            }
            Fe(l)
        }

        pub fn to_be_bytes(&self) -> [u8; 32] {
            let mut b = [0u8; 32];
            for i in 0..4 {
                b[8 * (3 - i)..8 * (3 - i) + 8].copy_from_slice(&self.0[i].to_be_bytes());
            }
            b
        }

        // reduce a raw 256-bit value into [0, p) — one conditional subtraction
        #[inline(always)]
        fn reduce_once(v: [u64; 4]) -> Fe {
            if v >= P {
                let mut r = v;
                let mut borrow = 0i128;
                for i in 0..4 {
                    let d = r[i] as i128 - P[i] as i128 - borrow;
                    if d < 0 {
                        r[i] = (d + (1i128 << 64)) as u64;
                        borrow = 1;
                    } else {
                        r[i] = d as u64;
                        borrow = 0;
                    }
                }
                Fe(r)
            } else {
                Fe(v)
            }
        }

        #[inline(always)]
        pub fn add(&self, r: &Fe) -> Fe {
            let mut t = [0u64; 4];
            let mut carry = 0u128;
            for i in 0..4 {
                let s = self.0[i] as u128 + r.0[i] as u128 + carry;
                t[i] = s as u64;
                carry = s >> 64;
            }
            if carry != 0 {
                // value = t + 2^256; value - p = t + C, which fits and is < p
                let mut s = t[0] as u128 + C;
                t[0] = s as u64;
                s >>= 64;
                for i in 1..4 {
                    s += t[i] as u128;
                    t[i] = s as u64;
                    s >>= 64;
                }
                debug_assert_eq!(s, 0);
                Fe(t)
            } else if t >= P {
                Fe::reduce_once(t)
            } else {
                Fe(t)
            }
        }

        #[inline(always)]
        pub fn sub(&self, r: &Fe) -> Fe {
            let mut t = [0u64; 4];
            let mut borrow = 0i128;
            for i in 0..4 {
                let d = self.0[i] as i128 - r.0[i] as i128 - borrow;
                if d < 0 {
                    t[i] = (d + (1i128 << 64)) as u64;
                    borrow = 1;
                } else {
                    t[i] = d as u64;
                    borrow = 0;
                }
            }
            if borrow != 0 {
                // value = t - 2^256; value + p = t - C, guaranteed in [0, p)
                let mut s = t[0] as i128 - C as i128;
                if s < 0 {
                    t[0] = (s + (1i128 << 64)) as u64;
                    s = 1;
                } else {
                    t[0] = s as u64;
                    s = 0;
                }
                let mut i = 1;
                while s != 0 && i < 4 {
                    let d = t[i] as i128 - s;
                    if d < 0 {
                        t[i] = (d + (1i128 << 64)) as u64;
                        s = 1;
                    } else {
                        t[i] = d as u64;
                        s = 0;
                    }
                    i += 1;
                }
                debug_assert_eq!(s, 0, "sub underflow");
                Fe(t)
            } else {
                Fe(t)
            }
        }

        // Schoolbook 4x4 -> 8 limbs, fully unrolled. Each product is folded
        // into a 64-bit low digit plus a u128 high accumulator (carries are
        // propagated immediately), which can never overflow 128 bits.
        // Invariant per column: total = lo + hi*2^64 exactly, hi < 2^66.
        #[inline(always)]
        pub fn mul(&self, r: &Fe) -> Fe {
            let a = self.0;
            let b = r.0;
            let (a0, a1, a2, a3) = (a[0] as u128, a[1] as u128, a[2] as u128, a[3] as u128);
            let (b0, b1, b2, b3) = (b[0] as u128, b[1] as u128, b[2] as u128, b[3] as u128);
            let mut t = [0u64; 8];
            let mut lo: u64 = 0;
            let mut hi: u128 = 0;
            macro_rules! acc {
                ($p:expr) => {{
                    let (nl, c) = lo.overflowing_add($p as u64);
                    lo = nl;
                    hi += ($p >> 64) + c as u128;
                }};
            }
            macro_rules! nextcol {
                ($idx:expr) => {{
                    t[$idx] = lo;
                    lo = hi as u64;
                    hi >>= 64;
                }};
            }
            acc!(a0 * b0);
            nextcol!(0);
            acc!(a0 * b1);
            acc!(a1 * b0);
            nextcol!(1);
            acc!(a0 * b2);
            acc!(a1 * b1);
            acc!(a2 * b0);
            nextcol!(2);
            acc!(a0 * b3);
            acc!(a1 * b2);
            acc!(a2 * b1);
            acc!(a3 * b0);
            nextcol!(3);
            acc!(a1 * b3);
            acc!(a2 * b2);
            acc!(a3 * b1);
            nextcol!(4);
            acc!(a2 * b3);
            acc!(a3 * b2);
            nextcol!(5);
            acc!(a3 * b3);
            nextcol!(6);
            t[7] = lo;
            debug_assert_eq!(hi, 0); // product < 2^512

            // fold 512 -> ~256 using 2^256 == C (mod p)
            let mut v = [0u64; 4];
            let mut c: u128 = 0;
            for i in 0..4 {
                let s = t[i] as u128 + (t[4 + i] as u128) * C + c;
                v[i] = s as u64;
                c = s >> 64; // < 2^34
            }
            // fold remaining high part (c < 2^34) times C; provably terminates
            while c != 0 {
                let s = v[0] as u128 + c * C;
                v[0] = s as u64;
                let mut cc = (s >> 64) as u64;
                let mut i = 1;
                while cc != 0 && i < 4 {
                    let s2 = v[i] as u128 + cc as u128;
                    v[i] = s2 as u64;
                    cc = (s2 >> 64) as u64;
                    i += 1;
                }
                c = cc as u128; // 0 or 1; if 1 the next round is the last
            }
            Fe::reduce_once(v)
        }

        // modular inverse via Fermat: a^(p-2). ~380 muls; called once per
        // 1024-key batch, so ~0.4 muls per key.
        pub fn inv(&self) -> Fe {
            let e: [u64; 4] = [0xFFFF_FFFE_FFFF_FC2D, !0, !0, !0]; // p - 2
            let mut acc = Fe::ONE;
            for bit in (0..256).rev() {
                acc = acc.mul(&acc);
                if (e[bit / 64] >> (bit % 64)) & 1 == 1 {
                    acc = acc.mul(self);
                }
            }
            acc
        }

        // used only by the startup self-test (independent reference)
        pub fn shr1(&self) -> Fe {
            let mut r = self.0;
            let mut carry = 0u64;
            for i in (0..4).rev() {
                let nc = r[i] & 1;
                r[i] >>= 1;
                r[i] |= carry << 63;
                carry = nc;
            }
            Fe(r)
        }
    }
}
use fe::Fe;

// ============================= affine points ================================

#[derive(Clone, Copy)]
struct Pt {
    x: Fe,
    y: Fe,
}

#[inline(always)]
fn prefix_of(y: &Fe) -> u8 {
    0x02 | (y.0[0] & 1) as u8
}

impl Pt {
    fn to_pub33(&self) -> [u8; 33] {
        let mut b = [0u8; 33];
        b[0] = prefix_of(&self.y);
        b[1..].copy_from_slice(&self.x.to_be_bytes());
        b
    }

    // single affine addition P + Q (P != +-Q); used for one-time table setup
    fn add_point(&self, q: &Pt) -> Pt {
        let dx = self.x.sub(&q.x);
        let dy = self.y.sub(&q.y);
        let lam = dy.mul(&dx.inv());
        let x3 = lam.mul(&lam).sub(&self.x).sub(&q.x);
        let y3 = lam.mul(&self.x.sub(&x3)).sub(&self.y);
        Pt { x: x3, y: y3 }
    }
}

// ========================= crate-based reference paths ======================

#[inline(always)]
fn key_bytes(k: u128) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[16..].copy_from_slice(&k.to_be_bytes());
    b
}

fn crate_pub33(k: u128) -> [u8; 33] {
    let secp = Secp256k1::signing_only();
    let sk = SecretKey::from_slice(&key_bytes(k)).expect("valid key");
    PublicKey::from_secret_key(&secp, &sk).serialize()
}

fn crate_point(k: u128) -> Pt {
    let secp = Secp256k1::signing_only();
    let sk = SecretKey::from_slice(&key_bytes(k)).expect("valid key");
    let pk = PublicKey::from_secret_key(&secp, &sk);
    let u = pk.serialize_uncompressed();
    assert_eq!(u[0], 0x04, "uncompressed pubkey");
    Pt {
        x: Fe::from_be_bytes(u[1..33].try_into().unwrap()),
        y: Fe::from_be_bytes(u[33..65].try_into().unwrap()),
    }
}

fn crate_hash160_of_pub33(pk: &[u8; 33]) -> [u8; 20] {
    let s: [u8; 32] = Sha256::digest(pk).into();
    Ripemd160::digest(s).into()
}

fn privkey_to_address(k: u128) -> String {
    let pk = crate_pub33(k);
    let h = crate_hash160_of_pub33(&pk);
    let mut v = Vec::with_capacity(21);
    v.push(0u8);
    v.extend_from_slice(&h);
    bs58::encode(v).with_check().into_string()
}

// ============================== SHA-256 =====================================
// We only ever hash exactly 33 bytes (a compressed public key), which is a
// single 64-byte block with a fixed padding/length tail.

const IV_SHA: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const K_SHA: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_33_scalar(prefix: u8, x: &Fe) -> [u8; 32] {
    let xb = x.to_be_bytes();
    // message = prefix || x (33 bytes); words 0..8 + fixed padding
    let mut w = [0u32; 64];
    w[0] = ((prefix as u32) << 24)
        | ((xb[0] as u32) << 16)
        | ((xb[1] as u32) << 8)
        | (xb[2] as u32);
    for i in 1..8 {
        let o = 4 * i - 1; // word i = message bytes 4i..4i+3 = xb[4i-1..4i+2]
        w[i] = u32::from_be_bytes([xb[o], xb[o + 1], xb[o + 2], xb[o + 3]]);
    }
    w[8] = ((xb[31] as u32) << 24) | 0x0080_0000; // last msg byte + 0x80 pad
    w[15] = 264; // 33 bytes * 8 bits

    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let (mut a, mut b, mut c, mut d) = (IV_SHA[0], IV_SHA[1], IV_SHA[2], IV_SHA[3]);
    let (mut e, mut f, mut g, mut h) = (IV_SHA[4], IV_SHA[5], IV_SHA[6], IV_SHA[7]);
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K_SHA[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    let mut out = [0u8; 32];
    let hs = [
        IV_SHA[0].wrapping_add(a),
        IV_SHA[1].wrapping_add(b),
        IV_SHA[2].wrapping_add(c),
        IV_SHA[3].wrapping_add(d),
        IV_SHA[4].wrapping_add(e),
        IV_SHA[5].wrapping_add(f),
        IV_SHA[6].wrapping_add(g),
        IV_SHA[7].wrapping_add(h),
    ];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&hs[i].to_be_bytes());
    }
    out
}

// SHA-NI path (canonical Intel sequence: 2 rnds2 per 4 rounds, msg1/msg2 for
// the schedule). Enabled at runtime via is_x86_feature_detected!("sha").
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha,sse4.1,ssse3")]
unsafe fn sha256_33_shani(prefix: u8, x: &Fe) -> [u8; 32] {
    let xb = x.to_be_bytes();
    let mut blk = [0u8; 64];
    blk[0] = prefix;
    blk[1..33].copy_from_slice(&xb);
    blk[33] = 0x80;
    blk[62] = 0x01; // 264 = 0x00000108 big-endian
    blk[63] = 0x08;

    let mask = _mm_set_epi64x(0x0c0d0e0f_08090a0b, 0x04050607_00010203);
    let mut s1 = _mm_loadu_si128(IV_SHA.as_ptr().add(4) as *const __m128i); // E F G H
    let t0 = _mm_loadu_si128(IV_SHA.as_ptr() as *const __m128i); // A B C D
    let t0 = _mm_shuffle_epi32(t0, 0xB1); // CDAB
    s1 = _mm_shuffle_epi32(s1, 0x1B); // HGFE
    let mut s0 = _mm_alignr_epi8(t0, s1, 8); // ABEF
    s1 = _mm_blend_epi16(s1, t0, 0xF0); // CDGH
    let abef0 = s0;
    let cdgh0 = s1;

    macro_rules! rnds {
        ($msg:expr) => {
            s1 = _mm_sha256rnds2_epu32(s1, s0, $msg);
            let t = _mm_alignr_epi8($msg, $msg, 8);
            s0 = _mm_sha256rnds2_epu32(s0, s1, t);
        };
    }
    #[inline(always)]
    unsafe fn kc(g: usize) -> __m128i {
        _mm_set_epi64x(
            (((K_SHA[4 * g + 3] as u64) << 32) | K_SHA[4 * g + 2] as u64) as i64,
            (((K_SHA[4 * g + 1] as u64) << 32) | K_SHA[4 * g] as u64) as i64,
        )
    }
    #[inline(always)]
    unsafe fn ld(blk: &[u8; 64], off: usize, mask: __m128i) -> __m128i {
        _mm_shuffle_epi8(_mm_loadu_si128(blk.as_ptr().add(off) as *const __m128i), mask)
    }

    let mut m = [_mm_setzero_si128(); 4];
    let mut msg;
    // rounds 0-15: load words, 2x rnds2 each group; prime msg1 partials
    m[0] = ld(&blk, 0, mask);
    msg = _mm_add_epi32(m[0], kc(0));
    rnds!(msg);
    m[1] = ld(&blk, 16, mask);
    msg = _mm_add_epi32(m[1], kc(1));
    rnds!(msg);
    m[0] = _mm_sha256msg1_epu32(m[0], m[1]);
    m[2] = ld(&blk, 32, mask);
    msg = _mm_add_epi32(m[2], kc(2));
    rnds!(msg);
    m[1] = _mm_sha256msg1_epu32(m[1], m[2]);
    m[3] = ld(&blk, 48, mask);
    msg = _mm_add_epi32(m[3], kc(3));
    rnds!(msg);
    let t = _mm_alignr_epi8(m[3], m[2], 4); // W9..W12
    m[0] = _mm_add_epi32(m[0], t);
    m[0] = _mm_sha256msg2_epu32(m[0], m[3]); // m0 = W16..W19
    m[2] = _mm_sha256msg1_epu32(m[2], m[3]); // partial W24..W27

    // rounds 16-63: repeating 4-group pattern
    macro_rules! group {
        ($g:expr) => {{
            let gi = $g % 4;
            let pi = ($g + 3) % 4;
            let fi = ($g + 1) % 4;
            msg = _mm_add_epi32(m[gi], kc($g));
            rnds!(msg);
            let t = _mm_alignr_epi8(m[gi], m[pi], 4); // W[n-7..n-4] terms
            m[fi] = _mm_add_epi32(m[fi], t);
            m[fi] = _mm_sha256msg2_epu32(m[fi], m[gi]);
            m[pi] = _mm_sha256msg1_epu32(m[pi], m[gi]);
        }};
    }
    group!(4);
    group!(5);
    group!(6);
    group!(7);
    group!(8);
    group!(9);
    group!(10);
    group!(11);
    group!(12);
    group!(13);
    group!(14);
    // rounds 60-63
    msg = _mm_add_epi32(m[15 % 4], kc(15));
    rnds!(msg);

    s0 = _mm_add_epi32(s0, abef0);
    s1 = _mm_add_epi32(s1, cdgh0);
    // s0 lanes [F,E,B,A] (ABEF), s1 lanes [H,G,D,C] (CDGH)
    let t = _mm_shuffle_epi32(s0, 0x1B); // [A, B, E, F]
    let u = _mm_shuffle_epi32(s1, 0xB1); // [G, H, C, D]
    let x = _mm_blend_epi16(t, u, 0xF0); // [A, B, C, D]
    let y = _mm_alignr_epi8(u, t, 8); // [E, F, G, H]
    let mut w = [0u32; 8];
    _mm_storeu_si128(w.as_mut_ptr() as *mut __m128i, x);
    _mm_storeu_si128(w.as_mut_ptr().add(4) as *mut __m128i, y);
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&w[i].to_be_bytes());
    }
    out
}

// ============================== RIPEMD-160 ==================================
// We only ever hash exactly 32 bytes (a SHA-256 digest): one block, fixed
// padding (X[8]=0x80, X[14]=256).

const RMD_R: [usize; 80] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, //
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8, //
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12, //
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2, //
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13,
];
const RMD_RR: [usize; 80] = [
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, //
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2, //
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13, //
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14, //
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11,
];
const RMD_S: [u32; 80] = [
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, //
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12, //
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5, //
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12, //
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6,
];
const RMD_SS: [u32; 80] = [
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, //
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11, //
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5, //
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8, //
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11,
];
const RMD_KL: [u32; 5] = [0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E];
const RMD_KR: [u32; 5] = [0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000];
const RMD_H: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];

#[inline(always)]
fn rmd_f(round: usize, x: u32, y: u32, z: u32) -> u32 {
    match round {
        0 => x ^ y ^ z,
        1 => (x & y) | (!x & z),
        2 => (x | !y) ^ z,
        3 => (x & z) | (y & !z),
        _ => x ^ (y | !z),
    }
}

fn ripemd160_32_scalar(inp: &[u8; 32]) -> [u8; 20] {
    let mut x = [0u32; 16];
    for i in 0..8 {
        x[i] = u32::from_le_bytes(inp[4 * i..4 * i + 4].try_into().unwrap());
    }
    x[8] = 0x80;
    x[14] = 256; // 32 bytes * 8 bits, little-endian length word

    let (mut al, mut bl, mut cl, mut dl, mut el) =
        (RMD_H[0], RMD_H[1], RMD_H[2], RMD_H[3], RMD_H[4]);
    let (mut ar, mut br, mut cr, mut dr, mut er) = (al, bl, cl, dl, el);

    for i in 0..80 {
        let rnd = i / 16;
        // left line
        let t = al
            .wrapping_add(rmd_f(rnd, bl, cl, dl))
            .wrapping_add(x[RMD_R[i]])
            .wrapping_add(RMD_KL[rnd])
            .rotate_left(RMD_S[i])
            .wrapping_add(el);
        let ncl = cl.rotate_left(10);
        al = el;
        el = dl;
        dl = ncl;
        cl = bl;
        bl = t;
        // right line (f and K' round order: f reverses, K' does not)
        let t2 = ar
            .wrapping_add(rmd_f(4 - rnd, br, cr, dr))
            .wrapping_add(x[RMD_RR[i]])
            .wrapping_add(RMD_KR[rnd])
            .rotate_left(RMD_SS[i])
            .wrapping_add(er);
        let ncr = cr.rotate_left(10);
        ar = er;
        er = dr;
        dr = ncr;
        cr = br;
        br = t2;
    }

    let h = [
        RMD_H[1].wrapping_add(cl).wrapping_add(dr),
        RMD_H[2].wrapping_add(dl).wrapping_add(er),
        RMD_H[3].wrapping_add(el).wrapping_add(ar),
        RMD_H[4].wrapping_add(al).wrapping_add(br),
        RMD_H[0].wrapping_add(bl).wrapping_add(cr),
    ];
    let mut out = [0u8; 20];
    for i in 0..5 {
        out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_le_bytes());
    }
    out
}

// -------- 8-way AVX2 RIPEMD-160 (one message per lane) --------

#[cfg(target_arch = "x86_64")]
struct RmdVecs {
    sv: Vec<__m256i>,  // broadcast S[i]  (left line rotations)
    nsv: Vec<__m256i>, // broadcast 32 - S[i]
    ss: Vec<__m256i>,  // broadcast SS[i] (right line rotations)
    nss: Vec<__m256i>, // broadcast 32 - SS[i]
    kl: Vec<__m256i>,  // broadcast left K
    kr: Vec<__m256i>,  // broadcast right K
    msgc: [__m256i; 16], // constant schedule words 8..15 (padding/length)
}

#[cfg(target_arch = "x86_64")]
static RMD_VECS: OnceLock<RmdVecs> = OnceLock::new();

#[cfg(target_arch = "x86_64")]
fn rmd_vecs() -> &'static RmdVecs {
    RMD_VECS.get_or_init(|| unsafe {
        let mut sv = Vec::with_capacity(80);
        let mut nsv = Vec::with_capacity(80);
        let mut ss = Vec::with_capacity(80);
        let mut nss = Vec::with_capacity(80);
        let mut kl = Vec::with_capacity(80);
        let mut kr = Vec::with_capacity(80);
        for i in 0..80 {
            sv.push(_mm256_set1_epi32(RMD_S[i] as i32));
            nsv.push(_mm256_set1_epi32((32 - RMD_S[i]) as i32));
            ss.push(_mm256_set1_epi32(RMD_SS[i] as i32));
            nss.push(_mm256_set1_epi32((32 - RMD_SS[i]) as i32));
            kl.push(_mm256_set1_epi32(RMD_KL[i / 16] as i32));
            kr.push(_mm256_set1_epi32(RMD_KR[i / 16] as i32));
        }
        let mut msgc = [_mm256_setzero_si256(); 16];
        msgc[8] = _mm256_set1_epi32(0x80);
        msgc[14] = _mm256_set1_epi32(256);
        RmdVecs { sv, nsv, ss, nss, kl, kr, msgc }
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ripemd8_avx2(ins: &[[u8; 32]; 8]) -> [__m256i; 5] {
    let rv = rmd_vecs();
    // transpose: m[j] lane g = word j of message g
    let mut m = rv.msgc;
    for j in 0..8 {
        let w = |g: usize| u32::from_le_bytes(ins[g][4 * j..4 * j + 4].try_into().unwrap());
        m[j] = _mm256_set_epi32(
            w(7) as i32, w(6) as i32, w(5) as i32, w(4) as i32,
            w(3) as i32, w(2) as i32, w(1) as i32, w(0) as i32,
        );
    }
    let v = |x: u32| _mm256_set1_epi32(x as i32);
    let mut al = v(RMD_H[0]);
    let mut bl = v(RMD_H[1]);
    let mut cl = v(RMD_H[2]);
    let mut dl = v(RMD_H[3]);
    let mut el = v(RMD_H[4]);
    let mut ar = al;
    let mut br = bl;
    let mut cr = cl;
    let mut dr = dl;
    let mut er = el;

    #[inline(always)]
    unsafe fn rol(x: __m256i, s: __m256i, ns: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_sllv_epi32(x, s), _mm256_srlv_epi32(x, ns))
    }
    #[inline(always)]
    unsafe fn rol10(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_slli_epi32(x, 10), _mm256_srli_epi32(x, 22))
    }
    #[inline(always)]
    unsafe fn f1(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
        _mm256_xor_si256(_mm256_xor_si256(x, y), z)
    }
    #[inline(always)]
    unsafe fn f2(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_and_si256(x, y), _mm256_andnot_si256(x, z))
    }
    #[inline(always)]
    unsafe fn f3(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
        _mm256_xor_si256(_mm256_or_si256(x, _mm256_xor_si256(y, _mm256_set1_epi32(-1))), z)
    }
    #[inline(always)]
    unsafe fn f4(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_and_si256(x, z), _mm256_andnot_si256(z, y))
    }
    #[inline(always)]
    unsafe fn f5(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
        _mm256_xor_si256(x, _mm256_or_si256(y, _mm256_xor_si256(z, _mm256_set1_epi32(-1))))
    }

    macro_rules! step {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $x:expr, $k:expr, $s:expr, $ns:expr) => {{
            let t = _mm256_add_epi32(
                _mm256_add_epi32($f($b, $c, $d), _mm256_add_epi32($x, $k)),
                $a,
            );
            $a = _mm256_add_epi32(rol(t, $s, $ns), $e);
            $c = rol10($c);
        }};
    }
    // note: register rotation (a,b,c,d,e) <- (e, T, b, rol10(c), d) is done
    // explicitly after each step pair via temporaries.

    for i in 0..80 {
        let rnd = i / 16;
        // left
        match rnd {
            0 => step!(al, bl, cl, dl, el, f1, m[RMD_R[i]], rv.kl[i], rv.sv[i], rv.nsv[i]),
            1 => step!(al, bl, cl, dl, el, f2, m[RMD_R[i]], rv.kl[i], rv.sv[i], rv.nsv[i]),
            2 => step!(al, bl, cl, dl, el, f3, m[RMD_R[i]], rv.kl[i], rv.sv[i], rv.nsv[i]),
            3 => step!(al, bl, cl, dl, el, f4, m[RMD_R[i]], rv.kl[i], rv.sv[i], rv.nsv[i]),
            _ => step!(al, bl, cl, dl, el, f5, m[RMD_R[i]], rv.kl[i], rv.sv[i], rv.nsv[i]),
        }
        // (a,b,c,d,e) <- (e, a, b, rol10(c), d)
        let (a0, b0, c0, d0) = (al, bl, cl, dl);
        al = el;
        el = d0;
        dl = c0;
        cl = b0;
        bl = a0;
        // right
        match 4 - rnd {
            0 => step!(ar, br, cr, dr, er, f1, m[RMD_RR[i]], rv.kr[i], rv.ss[i], rv.nss[i]),
            1 => step!(ar, br, cr, dr, er, f2, m[RMD_RR[i]], rv.kr[i], rv.ss[i], rv.nss[i]),
            2 => step!(ar, br, cr, dr, er, f3, m[RMD_RR[i]], rv.kr[i], rv.ss[i], rv.nss[i]),
            3 => step!(ar, br, cr, dr, er, f4, m[RMD_RR[i]], rv.kr[i], rv.ss[i], rv.nss[i]),
            _ => step!(ar, br, cr, dr, er, f5, m[RMD_RR[i]], rv.kr[i], rv.ss[i], rv.nss[i]),
        }
        let (a0, b0, c0, d0) = (ar, br, cr, dr);
        ar = er;
        er = d0;
        dr = c0;
        cr = b0;
        br = a0;
    }

    [
        _mm256_add_epi32(_mm256_add_epi32(v(RMD_H[1]), cl), dr),
        _mm256_add_epi32(_mm256_add_epi32(v(RMD_H[2]), dl), er),
        _mm256_add_epi32(_mm256_add_epi32(v(RMD_H[3]), el), ar),
        _mm256_add_epi32(_mm256_add_epi32(v(RMD_H[4]), al), br),
        _mm256_add_epi32(_mm256_add_epi32(v(RMD_H[0]), bl), cr),
    ]
}

// ========================= batch point addition =============================
// Montgomery's trick: n affine point additions share ONE modular inversion.

struct Scratch {
    xs: Vec<Fe>,
    ys: Vec<Fe>,
    dxs: Vec<Fe>,
    pre: Vec<Fe>,
    sha8: [[u8; 32]; 8],
}

impl Scratch {
    fn new(b: usize) -> Scratch {
        Scratch {
            xs: vec![Fe::ZERO; b],
            ys: vec![Fe::ZERO; b],
            dxs: vec![Fe::ZERO; b],
            pre: vec![Fe::ZERO; b],
            sha8: [[0u8; 32]; 8],
        }
    }
}

// one affine addition P1 + Q given 1/(x1 - x2); requires P1 != +-Q
#[inline(always)]
fn finish_add(x1: &Fe, y1: &Fe, qx: &Fe, qy: &Fe, inv_dx: Fe) -> (Fe, Fe) {
    let dy = y1.sub(qy);
    let lam = dy.mul(&inv_dx);
    let x3 = lam.mul(&lam).sub(x1).sub(qx);
    let y3 = lam.mul(&x1.sub(&x3)).sub(y1);
    (x3, y3)
}

// add the same `step` point to all n strand points, one shared inversion.
// Structured as: (1) deltas, (2) prefix-product chain (latency-bound),
// (3) backward inversion chain turning pre[i] into 1/dx[i] (latency-bound),
// (4) independent point updates (throughput-bound, no bounds checks).
fn batch_add_round(sc: &mut Scratch, step: &Pt) -> bool {
    let n = sc.xs.len();
    {
        let (xs, dxs) = (&sc.xs, &mut sc.dxs);
        let mut i = 0;
        while i < n {
            dxs[i] = xs[i].sub(&step.x);
            if dxs[i].is_zero() {
                return false; // P == +-Q: cannot happen for in-range keys; skip chunk
            }
            i += 1;
        }
    }
    {
        let (dxs, pre) = (&sc.dxs, &mut sc.pre);
        let mut acc = dxs[0];
        pre[0] = acc;
        let mut i = 1;
        while i < n {
            acc = acc.mul(&dxs[i]);
            pre[i] = acc;
            i += 1;
        }
    }
    {
        let (dxs, pre) = (&sc.dxs, &mut sc.pre);
        let mut inv = pre[n - 1].inv();
        let mut i = n - 1;
        while i >= 1 {
            let inv_dxi = inv.mul(&pre[i - 1]);
            pre[i] = inv_dxi;
            inv = inv.mul(&dxs[i]);
            i -= 1;
        }
        pre[0] = inv;
    }
    {
        let (xs, ys, pre) = (&mut sc.xs, &mut sc.ys, &sc.pre);
        let (sx, sy) = (step.x, step.y);
        for ((x, y), inv) in xs.iter_mut().zip(ys.iter_mut()).zip(pre.iter()) {
            let (x3, y3) = finish_add(x, y, &sx, &sy, *inv);
            *x = x3;
            *y = y3;
        }
    }
    true
}

// strand init: strand 0 = base; strand i (i>=1) = base + gtable[i-1]
fn strand_init(sc: &mut Scratch, base: &Pt, gtable: &[Pt]) -> bool {
    let n = sc.xs.len();
    sc.xs[0] = base.x;
    sc.ys[0] = base.y;
    if n == 1 {
        return true;
    }
    let m = n - 1;
    for i in 1..n {
        sc.xs[i] = base.x;
        sc.ys[i] = base.y;
    }
    for j in 0..m {
        sc.dxs[j] = base.x.sub(&gtable[j].x);
        if sc.dxs[j].is_zero() {
            return false;
        }
    }
    sc.pre[0] = sc.dxs[0];
    for j in 1..m {
        sc.pre[j] = sc.pre[j - 1].mul(&sc.dxs[j]);
    }
    let mut inv = sc.pre[m - 1].inv();
    for j in (1..m).rev() {
        let inv_dxj = inv.mul(&sc.pre[j - 1]);
        let (x3, y3) = finish_add(
            &sc.xs[j + 1],
            &sc.ys[j + 1],
            &gtable[j].x,
            &gtable[j].y,
            inv_dxj,
        );
        sc.xs[j + 1] = x3;
        sc.ys[j + 1] = y3;
        inv = inv.mul(&sc.dxs[j]);
    }
    let (x3, y3) = finish_add(&sc.xs[1], &sc.ys[1], &gtable[0].x, &gtable[0].y, inv);
    sc.xs[1] = x3;
    sc.ys[1] = y3;
    true
}

// ============================ hash dispatch =================================

#[derive(Clone, Copy)]
struct HashCfg {
    shani: bool,
    rmd_avx2: bool,
}

impl HashCfg {
    fn detect() -> HashCfg {
        let force = std::env::var("KEYSEARCH_SCALAR").is_ok();
        let mut c = HashCfg {
            shani: false,
            rmd_avx2: false,
        };
        #[cfg(target_arch = "x86_64")]
        {
            if !force && is_x86_feature_detected!("sha") {
                c.shani = true;
            }
            if !force && is_x86_feature_detected!("avx2") {
                c.rmd_avx2 = true;
            }
        }
        c
    }

    #[inline(always)]
    fn sha256_33(&self, prefix: u8, x: &Fe) -> [u8; 32] {
        #[cfg(target_arch = "x86_64")]
        {
            if self.shani {
                return unsafe { sha256_33_shani(prefix, x) };
            }
        }
        sha256_33_scalar(prefix, x)
    }
}

// hash all n strand points; on a first-word match the key is fully re-derived
// with the crates before being reported (no SIMD result is ever trusted)
fn hash_round(sc: &mut Scratch, key_base: u128, target: &[u8; 20], cfg: HashCfg) -> Option<u128> {
    let n = sc.xs.len();
    let mut i = 0usize;
    #[cfg(target_arch = "x86_64")]
    {
        if cfg.rmd_avx2 {
            let tw = u32::from_le_bytes([target[0], target[1], target[2], target[3]]);
            while i + 8 <= n {
                for g in 0..8 {
                    sc.sha8[g] = cfg.sha256_33(prefix_of(&sc.ys[i + g]), &sc.xs[i + g]);
                }
                unsafe {
                    let hs = ripemd8_avx2(&sc.sha8);
                    let mut words = [0u32; 8];
                    _mm256_storeu_si256(words.as_mut_ptr() as *mut __m256i, hs[0]);
                    for lane in 0..8 {
                        if words[lane] == tw {
                            let k = key_base + (i + lane) as u128;
                            if crate_hash160_of_pub33(&crate_pub33(k)) == *target {
                                return Some(k);
                            }
                            eprintln!("note: hash false positive at {:x} (ignored)", k);
                        }
                    }
                }
                i += 8;
            }
        }
    }
    while i < n {
        let s = cfg.sha256_33(prefix_of(&sc.ys[i]), &sc.xs[i]);
        if ripemd160_32_scalar(&s) == *target {
            let k = key_base + i as u128;
            if crate_hash160_of_pub33(&crate_pub33(k)) == *target {
                return Some(k);
            }
        }
        i += 1;
    }
    None
}

// per-chunk sanity: push the first 8 strand pubkeys through the ACTIVE hash
// backends and compare against the sha2/ripemd crates
fn verify_hash_pipeline(sc: &Scratch, cfg: HashCfg) -> bool {
    let n = sc.xs.len().min(8);
    let mut pks = [[0u8; 33]; 8];
    let mut shas = [[0u8; 32]; 8];
    for g in 0..n {
        pks[g][0] = prefix_of(&sc.ys[g]);
        pks[g][1..].copy_from_slice(&sc.xs[g].to_be_bytes());
        shas[g] = cfg.sha256_33(pks[g][0], &sc.xs[g]);
        let expect: [u8; 32] = Sha256::digest(pks[g]).into();
        if shas[g] != expect {
            return false;
        }
        let expect160: [u8; 20] = {
            let s: [u8; 32] = expect;
            Ripemd160::digest(s).into()
        };
        if ripemd160_32_scalar(&shas[g]) != expect160 {
            return false;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if cfg.rmd_avx2 && n == 8 {
            unsafe {
                let hs = ripemd8_avx2(&shas);
                let mut w = [[0u32; 8]; 5];
                for j in 0..5 {
                    _mm256_storeu_si256(w[j].as_mut_ptr() as *mut __m256i, hs[j]);
                }
                for g in 0..8 {
                    let mut h = [0u8; 20];
                    for j in 0..5 {
                        h[4 * j..4 * j + 4].copy_from_slice(&w[j][g].to_le_bytes());
                    }
                    if h != crate_hash160_of_pub33(&pks[g]) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

// ============================== search ======================================

enum ChunkResult {
    Found(u128),
    Stopped,
    Next,
}

struct SearchCtx<'a> {
    target: &'a [u8; 20],
    cfg: HashCfg,
    gtable: &'a [Pt],
    stop: &'a AtomicBool,
    counter: &'a AtomicU64,
}

// Covers keys [k0, k0 + b*rounds): strand j of round t is key k0 + j + t*b.
fn search_chunk(
    sc: &mut Scratch,
    k0: u128,
    rounds: u64,
    step: &Pt,
    ctx: &SearchCtx,
    verify_points: bool,
) -> ChunkResult {
    let b = sc.xs.len();
    let base = crate_point(k0);
    if !strand_init(sc, &base, ctx.gtable) {
        return ChunkResult::Next; // degenerate point collision: pick a new chunk
    }
    if !verify_hash_pipeline(sc, ctx.cfg) {
        die("runtime hash verification failed (CPU/SDK inconsistency) — aborting");
    }
    if verify_points {
        for j in [0usize, 1, b / 2, b - 1] {
            let expect = crate_point(k0 + j as u128);
            if sc.xs[j] != expect.x || sc.ys[j] != expect.y {
                die("runtime point verification failed — aborting");
            }
        }
    }
    if let Some(k) = hash_round(sc, k0, ctx.target, ctx.cfg) {
        return ChunkResult::Found(k);
    }
    ctx.counter.fetch_add(b as u64, Ordering::Relaxed);
    for t in 1..rounds {
        if ctx.stop.load(Ordering::Relaxed) {
            return ChunkResult::Stopped;
        }
        if !batch_add_round(sc, step) {
            return ChunkResult::Next;
        }
        let kb = k0 + t as u128 * b as u128;
        if let Some(k) = hash_round(sc, kb, ctx.target, ctx.cfg) {
            return ChunkResult::Found(k);
        }
        ctx.counter.fetch_add(b as u64, Ordering::Relaxed);
    }
    ChunkResult::Next
}

// FIX #7/#8: persist immediately from the worker, verify, survive poisoned mutex.
fn record_found(k: u128, found: &Mutex<Option<u128>>, stop: &AtomicBool) {
    let addr = privkey_to_address(k);
    if addr != TARGET_ADDRESS {
        eprintln!(
            "WARNING: derived address {} != target (recording anyway)",
            addr
        );
    }
    let line = format!("{} {:064x}\n", addr, k);
    eprintln!("\nFOUND: {}", line.trim());
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("FOUND.txt") {
        let _ = f.write_all(line.as_bytes());
        let _ = f.sync_all();
    }
    *found.lock().unwrap_or_else(|e| e.into_inner()) = Some(k);
    stop.store(true, Ordering::SeqCst);
}

fn worker(
    idx: usize,
    lo: u128,
    hi: u128,
    counters: Arc<Vec<Counter>>,
    stop: Arc<AtomicBool>,
    found: Arc<Mutex<Option<u128>>>,
    cfg: HashCfg,
    gtable: Arc<Vec<Pt>>,
) {
    if let Some(ids) = core_affinity::get_core_ids() {
        if !ids.is_empty() {
            let _ = core_affinity::set_for_current(ids[idx % ids.len()]);
        }
    }
    let span = hi - lo + 1;
    let b = (STRANDS as u128).min(span) as usize;
    let rounds = ((span / b as u128).min(ROUNDS as u128).max(1)) as u64;
    let chunk_keys = b as u128 * rounds as u128;
    let step = crate_point(b as u128);
    let ctx = SearchCtx {
        target: &TARGET_HASH160,
        cfg,
        gtable: &gtable,
        stop: &stop,
        counter: &counters[idx].0,
    };
    let mut sc = Scratch::new(b);
    let mut rng = rand::thread_rng();
    let mut chunk_no: u64 = 0;
    while !stop.load(Ordering::Relaxed) {
        let k0 = lo + rng.gen_range(0..=(span - chunk_keys));
        match search_chunk(&mut sc, k0, rounds, &step, &ctx, chunk_no % 256 == 0) {
            ChunkResult::Found(k) => {
                record_found(k, &found, &stop);
                return;
            }
            ChunkResult::Stopped => return,
            ChunkResult::Next => chunk_no += 1,
        }
    }
}

// ============================== self-tests ==================================

fn fe_random(rng: &mut impl Rng) -> Fe {
    loop {
        let mut l = [0u64; 4];
        for i in 0..4 {
            l[i] = rng.gen();
        }
        if l < fe::P {
            return Fe(l);
        }
    }
}

// independent mul reference: double-and-add using ONLY add/shift (no mul)
fn mul_reference(a: &Fe, b: &Fe) -> Fe {
    let mut acc = Fe::ZERO;
    let mut x = *a;
    let mut y = *b;
    for _ in 0..256 {
        if y.0[0] & 1 == 1 {
            acc = acc.add(&x);
        }
        x = x.add(&x);
        y = y.shr1();
    }
    acc
}

fn fermat_is_one(a: &Fe) -> bool {
    let e: [u64; 4] = [0xFFFF_FFFE_FFFF_FC2E, !0, !0, !0]; // p - 1
    let mut acc = Fe::ONE;
    for bit in (0..256).rev() {
        acc = acc.mul(&acc);
        if (e[bit / 64] >> (bit % 64)) & 1 == 1 {
            acc = acc.mul(a);
        }
    }
    acc == Fe::ONE
}

fn build_gtable() -> Vec<Pt> {
    // gtable[j] = (j+1)*G for j in 0..STRANDS-1.
    // The chain starts at 2*G because P+Q requires P != Q (1*G + 1*G would
    // be a doubling, which the affine formula cannot handle).
    let g = crate_point(1);
    let mut table = Vec::with_capacity(STRANDS - 1);
    table.push(g);
    let mut acc = crate_point(2);
    table.push(acc);
    for _ in 2..STRANDS - 1 {
        acc = acc.add_point(&g);
        table.push(acc);
    }
    for j in [0usize, 1, 7, 512, STRANDS - 2] {
        let expect = crate_point(j as u128 + 1);
        if table[j].x != expect.x || table[j].y != expect.y {
            die("self-test failed: G table");
        }
    }
    table
}

fn self_test(cfg: &mut HashCfg) -> Vec<Pt> {
    let mut rng = rand::thread_rng();

    // known address + target address/hash160 consistency
    if privkey_to_address(1) != "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH" {
        die("self-test failed: address derivation");
    }
    let decoded = match bs58::decode(TARGET_ADDRESS).with_check(None).into_vec() {
        Ok(v) => v,
        Err(_) => die("self-test failed: target address decode"),
    };
    if decoded.len() != 21 || decoded[0] != 0 || decoded[1..] != TARGET_HASH160[..] {
        die("self-test failed: TARGET_HASH160 does not match TARGET_ADDRESS");
    }

    // field arithmetic
    for _ in 0..32 {
        let a = fe_random(&mut rng);
        let b = fe_random(&mut rng);
        if a.sub(&b).add(&b) != a {
            die("self-test failed: Fe sub/add roundtrip");
        }
    }
    for _ in 0..8 {
        let a = fe_random(&mut rng);
        if Fe::ZERO.sub(&a).add(&a) != Fe::ZERO {
            die("self-test failed: Fe negation");
        }
    }
    for _ in 0..6 {
        let a = fe_random(&mut rng);
        let b = fe_random(&mut rng);
        if a.mul(&b) != mul_reference(&a, &b) {
            die("self-test failed: Fe mul vs double-and-add reference");
        }
    }
    let pm1 = Fe::ZERO.sub(&Fe::ONE); // p - 1 == -1
    if pm1.mul(&pm1) != Fe::ONE
        || Fe::ZERO.mul(&pm1) != Fe::ZERO
        || Fe::ONE.mul(&pm1) != pm1
    {
        die("self-test failed: Fe mul identities");
    }
    for _ in 0..3 {
        let a = fe_random(&mut rng);
        if !a.is_zero() && !fermat_is_one(&a) {
            die("self-test failed: Fermat little theorem");
        }
    }
    for _ in 0..4 {
        let a = fe_random(&mut rng);
        if !a.is_zero() && a.mul(&a.inv()) != Fe::ONE {
            die("self-test failed: Fe inv");
        }
    }

    // hash implementations vs crates (scalar must pass; SIMD falls back)
    for _ in 0..256 {
        let x = fe_random(&mut rng);
        let pfx: u8 = if rng.gen_bool(0.5) { 0x02 } else { 0x03 };
        let mut msg = [0u8; 33];
        msg[0] = pfx;
        msg[1..].copy_from_slice(&x.to_be_bytes());
        let expect: [u8; 32] = Sha256::digest(msg).into();
        if sha256_33_scalar(pfx, &x) != expect {
            die("self-test failed: scalar sha256");
        }
        #[cfg(target_arch = "x86_64")]
        {
            if cfg.shani {
                if unsafe { sha256_33_shani(pfx, &x) } != expect {
                    eprintln!("warning: SHA-NI self-test failed; using scalar SHA-256");
                    cfg.shani = false;
                }
            }
        }
        let expect160: [u8; 20] = {
            let s: [u8; 32] = expect;
            Ripemd160::digest(s).into()
        };
        if ripemd160_32_scalar(&expect) != expect160 {
            die("self-test failed: scalar ripemd160");
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if cfg.rmd_avx2 {
            let mut ok = true;
            for _ in 0..32 {
                let mut ins = [[0u8; 32]; 8];
                let mut exps = [[0u8; 20]; 8];
                for g in 0..8 {
                    ins[g] = rng.gen::<[u8; 32]>();
                    let d: [u8; 20] = Ripemd160::digest(ins[g]).into();
                    exps[g] = d;
                }
                unsafe {
                    let hs = ripemd8_avx2(&ins);
                    let mut w = [[0u32; 8]; 5];
                    for j in 0..5 {
                        _mm256_storeu_si256(w[j].as_mut_ptr() as *mut __m256i, hs[j]);
                    }
                    for g in 0..8 {
                        let mut h = [0u8; 20];
                        for j in 0..5 {
                            h[4 * j..4 * j + 4].copy_from_slice(&w[j][g].to_le_bytes());
                        }
                        if h != exps[g] {
                            ok = false;
                        }
                    }
                }
            }
            if !ok {
                eprintln!("warning: AVX2 RIPEMD-160 self-test failed; using scalar RIPEMD-160");
                cfg.rmd_avx2 = false;
            }
        }
    }

    // strand/batch point machinery vs crate scalar multiplication
    let gtable = build_gtable();
    {
        let k0 = RANGE_START + rng.gen_range(0..1u128 << 40);
        let b = STRANDS;
        let mut sc = Scratch::new(b);
        let base = crate_point(k0);
        if !strand_init(&mut sc, &base, &gtable) {
            die("self-test failed: strand init guard");
        }
        let step = crate_point(b as u128);
        let check = |sc: &Scratch, t: u64| {
            for j in [0usize, 1, 511, 1023] {
                let k = k0 + j as u128 + t as u128 * b as u128;
                let expect = crate_point(k);
                if sc.xs[j] != expect.x || sc.ys[j] != expect.y {
                    die("self-test failed: batch point addition");
                }
                let got = Pt {
                    x: sc.xs[j],
                    y: sc.ys[j],
                }
                .to_pub33();
                if got != crate_pub33(k) {
                    die("self-test failed: pubkey serialization");
                }
            }
        };
        check(&sc, 0);
        for t in 1..4 {
            if !batch_add_round(&mut sc, &step) {
                die("self-test failed: batch add guard");
            }
            check(&sc, t);
        }
    }

    // end-to-end planted-key search through the full pipeline
    // (active backends, then forced-scalar)
    for use_cfg in [*cfg, HashCfg { shani: false, rmd_avx2: false }] {
        let lo = RANGE_START;
        let planted = RANGE_START + 3000;
        let target = crate_hash160_of_pub33(&crate_pub33(planted));
        let stop = AtomicBool::new(false);
        let counter = AtomicU64::new(0);
        let b = STRANDS;
        let step = crate_point(b as u128);
        let ctx = SearchCtx {
            target: &target,
            cfg: use_cfg,
            gtable: &gtable,
            stop: &stop,
            counter: &counter,
        };
        let mut sc = Scratch::new(b);
        match search_chunk(&mut sc, lo, 4, &step, &ctx, true) {
            ChunkResult::Found(k) if k == planted => {}
            _ => die("self-test failed: planted-key search"),
        }
    }

    gtable
}

// ================================ main ======================================

// FIX #2: strict argument parsing, no silent fallback on typos.
fn parse_arg<T: FromStr>(args: &[String], i: usize, name: &str, default: T) -> T {
    match args.get(i) {
        None => default,
        Some(s) => s
            .parse()
            .unwrap_or_else(|_| die(&format!("invalid {}: {:?}", name, s))),
    }
}

// FIX #3: reject NaN/inf, empty windows, clamp upper bound, no underflow.
fn compute_window(pct: f64, start_pct: f64) -> (u128, u128) {
    if !pct.is_finite() || !start_pct.is_finite() || pct <= 0.0 || pct > 100.0 {
        die("range percent must be in (0, 100]");
    }
    if start_pct < 0.0 || start_pct + pct > 100.0 + 1e-9 {
        die("window start + range percent must be <= 100");
    }
    let total = RANGE_END - RANGE_START + 1;
    let a = (start_pct * 100_000.0).round() as u128;
    let b = (((start_pct + pct) * 100_000.0).round() as u128).min(SCALE);
    if b <= a {
        die("window is empty after rounding (resolution is 0.00001%)");
    }
    let lo = RANGE_START + total * a / SCALE;
    let hi = RANGE_START + total * b / SCALE - 1;
    (lo, hi)
}

fn onoff(b: bool) -> &'static str {
    if b {
        "fast"
    } else {
        "scalar"
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pct: f64 = parse_arg(&args, 1, "percent", RANGE_PARAM);
    let start_pct: f64 = parse_arg(&args, 2, "start percent", WINDOW_START_PCT);
    let default_workers = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let workers: usize = parse_arg(&args, 3, "workers", default_workers);
    if workers == 0 {
        die("workers must be >= 1");
    }

    let mut cfg = HashCfg::detect();
    let gtable = Arc::new(self_test(&mut cfg));

    if std::env::var("KEYSEARCH_TEST").is_ok() {
        println!(
            "All self-tests passed. backends: sha256={} ripemd160={}",
            onoff(cfg.shani),
            onoff(cfg.rmd_avx2)
        );
        return;
    }

    let (lo, hi) = compute_window(pct, start_pct);
    let span = hi - lo + 1;
    println!("Target  : {}", TARGET_ADDRESS);
    println!("Window  : {:x} - {:x} ({}% from {}%)", lo, hi, pct, start_pct);
    println!("Keys    : {} (~2^{:.1})", span, (span as f64).log2());
    println!(
        "Backend : sha256={} ripemd160={} | strands={} (batch inversion)",
        onoff(cfg.shani),
        onoff(cfg.rmd_avx2),
        STRANDS
    );
    println!("Workers : {}  chunk={}  (Ctrl+C to stop)\n", workers, 1usize << 20);

    let stop = Arc::new(AtomicBool::new(false));
    let found: Arc<Mutex<Option<u128>>> = Arc::new(Mutex::new(None));
    let counters: Arc<Vec<Counter>> =
        Arc::new((0..workers).map(|_| Counter(AtomicU64::new(0))).collect());

    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
            .expect("failed to set Ctrl+C handler");
    }

    let handles: Vec<_> = (0..workers)
        .map(|i| {
            let (c, s, f, g) = (counters.clone(), stop.clone(), found.clone(), gtable.clone());
            thread::Builder::new()
                .name(format!("worker-{}", i))
                .spawn(move || worker(i, lo, hi, c, s, f, cfg, g))
                .expect("failed to spawn worker")
        })
        .collect();

    let t0 = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(500));
        let n: u64 = counters.iter().map(|c| c.0.load(Ordering::Relaxed)).sum();
        let el = t0.elapsed().as_secs_f64().max(1e-9);
        print!(
            "\rTried {} | {:.2}M keys/s | {:.0}s   ",
            n,
            n as f64 / el / 1e6,
            el
        );
        std::io::stdout().flush().ok();
    }

    for h in handles {
        h.join().ok();
    }

    let total: u64 = counters.iter().map(|c| c.0.load(Ordering::Relaxed)).sum();
    let result = *found.lock().unwrap_or_else(|e| e.into_inner());
    match result {
        Some(k) => println!(
            "\n\nFOUND\nPrivate key: {:064x}\nAddress    : {}\n(saved to FOUND.txt)",
            k,
            privkey_to_address(k)
        ),
        None => println!("\n\nStopped. Total tried: {}", total),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    #[test]
    fn rmd_scalar_matches_crate() {
        let mut rng = rand::thread_rng();
        for _ in 0..128 {
            let inp: [u8; 32] = rng.gen();
            let expect: [u8; 20] = Ripemd160::digest(inp).into();
            assert_eq!(ripemd160_32_scalar(&inp), expect);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rmd_avx2_matches_crate() {
        if !is_x86_feature_detected!("avx2") { return; }
        let mut rng = rand::thread_rng();
        for _ in 0..64 {
            let mut ins = [[0u8; 32]; 8];
            for g in 0..8 { ins[g] = rng.gen(); }
            let hs = unsafe { ripemd8_avx2(&ins) };
            let mut w = [[0u32; 8]; 5];
            for j in 0..5 {
                unsafe { _mm256_storeu_si256(w[j].as_mut_ptr() as *mut __m256i, hs[j]) };
            }
            for g in 0..8 {
                let mut h = [0u8; 20];
                for j in 0..5 { h[4*j..4*j+4].copy_from_slice(&w[j][g].to_le_bytes()); }
                let expect: [u8; 20] = Ripemd160::digest(ins[g]).into();
                assert_eq!(h, expect, "lane {} input {}", g, hex(&ins[g]));
            }
        }
    }



    #[cfg(target_arch = "x86_64")]
    #[test]
    fn shani_matches_crate() {
        if !is_x86_feature_detected!("sha") { return; }
        let mut rng = rand::thread_rng();
        for _ in 0..256 {
            let x = fe_random(&mut rng);
            let pfx: u8 = if rng.gen_bool(0.5) { 0x02 } else { 0x03 };
            let mut msg = [0u8; 33];
            msg[0] = pfx;
            msg[1..].copy_from_slice(&x.to_be_bytes());
            let expect: [u8; 32] = Sha256::digest(msg).into();
            assert_eq!(unsafe { sha256_33_shani(pfx, &x) }, expect);
        }
    }

}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    #[test]
    fn components() {
        let mut rng = rand::thread_rng();
        let a = fe_random(&mut rng);
        let b = fe_random(&mut rng);
        let n = 200_000;
        let t = Instant::now();
        let mut acc = a;
        for _ in 0..n { acc = acc.mul(&b); }
        println!("Fe::mul      : {:>7.1} ns", (t.elapsed().as_secs_f64()*1e9)/(n as f64));
        let t = Instant::now();
        let mut acc2 = a;
        for _ in 0..2000 { acc2 = acc2.inv(); }
        println!("Fe::inv      : {:>7.1} ns", (t.elapsed().as_secs_f64()*1e9)/2000.0);
        let x = fe_random(&mut rng);
        let t = Instant::now();
        let mut h = [0u8;32];
        for _ in 0..n { h = sha256_33_scalar(2, &x); }
        println!("sha scalar   : {:>7.1} ns", (t.elapsed().as_secs_f64()*1e9)/(n as f64));
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sha") {
                // varying inputs (defeat CSE), result consumed via println
                let t = Instant::now();
                let mut acc = 0u32;
                for i in 0..n {
                    let xi = Fe([(i as u64) ^ 0x1234, (i as u64).wrapping_mul(3), 0xdeadbeef, i as u64]);
                    let v = unsafe { sha256_33_shani(2, &xi) };
                    acc = acc.wrapping_add(u32::from_le_bytes([v[0], v[1], v[2], v[3]]));
                }
                println!("sha shani    : {:>7.1} ns ({:08x})", (t.elapsed().as_secs_f64()*1e9)/(n as f64), acc);
                let t = Instant::now();
                let mut acc2 = 0u32;
                for i in 0..n {
                    let xi = Fe([(i as u64) ^ 0x1234, (i as u64).wrapping_mul(3), 0xdeadbeef, i as u64]);
                    let v = sha256_33_scalar(2, &xi);
                    acc2 = acc2.wrapping_add(u32::from_le_bytes([v[0], v[1], v[2], v[3]]));
                }
                println!("sha scalar-v  : {:>7.1} ns ({:08x})", (t.elapsed().as_secs_f64()*1e9)/(n as f64), acc2);
            }
        }
        let t = Instant::now();
        let mut r = [0u8;20];
        for _ in 0..n { r = ripemd160_32_scalar(&h); }
        println!("rmd scalar   : {:>7.1} ns", (t.elapsed().as_secs_f64()*1e9)/(n as f64));
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let ins = [h; 8];
                let t = Instant::now();
                let mut rr = [[0u8;32];8];
                for _ in 0..(n/8) { let _ = unsafe { ripemd8_avx2(&ins) }; rr[0] = ins[0]; }
                println!("rmd avx2x8   : {:>7.1} ns /key", (t.elapsed().as_secs_f64()*1e9)/(n as f64/8.0));
            }
        }
        // batch add round of 1024
        let gtable = build_gtable();
        let mut sc = Scratch::new(STRANDS);
        let base = crate_point(0x40_0000_0000_0000_0000);
        strand_init(&mut sc, &base, &gtable);
        let step = crate_point(STRANDS as u128);
        let t = Instant::now();
        let rounds = 2000;
        for _ in 0..rounds { if !batch_add_round(&mut sc, &step) { panic!() } }
        let el = t.elapsed().as_secs_f64();
        println!("batch_add    : {:>7.1} ns /key (1024-lane)", (el*1e9)/(rounds as f64*1024.0));
        // bisect: sha only, directly on scratch buffers
        {
            let mut sch = Scratch::new(STRANDS);
            strand_init(&mut sch, &base, &gtable);
            let cfg = HashCfg { shani: true, rmd_avx2: false };
            let mut acc = 0u32;
            let rounds = 500;
            let t = Instant::now();
            for _ in 0..rounds {
                for g in 0..STRANDS {
                    let v = cfg.sha256_33(prefix_of(&sch.ys[g]), &sch.xs[g]);
                    acc = acc.wrapping_add(v[0] as u32);
                }
            }
            let el = t.elapsed().as_secs_f64();
            println!("sha-only via cfg on scratch: {:>7.1} ns /key ({:08x})", (el*1e9)/(rounds as f64*1024.0), acc);
            // same but direct unsafe call
            let mut acc2 = 0u32;
            let t = Instant::now();
            for _ in 0..rounds {
                for g in 0..STRANDS {
                    let v = unsafe { sha256_33_shani(prefix_of(&sch.ys[g]), &sch.xs[g]) };
                    acc2 = acc2.wrapping_add(v[0] as u32);
                }
            }
            let el = t.elapsed().as_secs_f64();
            println!("sha-only direct on scratch: {:>7.1} ns /key ({:08x})", (el*1e9)/(rounds as f64*1024.0), acc2);
        }
        // inline combo: shani + scalar rmd (hash_round tail structure)
        {
            let mut sch = Scratch::new(STRANDS);
            strand_init(&mut sch, &base, &gtable);
            let target = TARGET_HASH160;
            let cfg = HashCfg { shani: true, rmd_avx2: false };
            let mut acc = 0u32;
            let rounds = 500;
            let t = Instant::now();
            for _ in 0..rounds {
                for g in 0..STRANDS {
                    let sv = cfg.sha256_33(prefix_of(&sch.ys[g]), &sch.xs[g]);
                    let hv = ripemd160_32_scalar(&sv);
                    if hv == target { panic!() }
                    acc = acc.wrapping_add(hv[0] as u32);
                }
            }
            let el = t.elapsed().as_secs_f64();
            println!("inline shani+rmdsc : {:>7.1} ns /key ({:08x})", (el*1e9)/(rounds as f64*1024.0), acc);
        }
        // hash_round alone (1024 keys, many rounds), all backend combos
        for (shani, rmda) in [(true, true), (false, true), (true, false), (false, false)] {
            let cfg = HashCfg { shani, rmd_avx2: rmda };
            let target = TARGET_HASH160;
            let mut sch = Scratch::new(STRANDS);
            strand_init(&mut sch, &base, &gtable);
            let rounds = 1000;
            let t = Instant::now();
            for _ in 0..rounds { if hash_round(&mut sch, 0, &target, cfg).is_some() { panic!() } }
            let el = t.elapsed().as_secs_f64();
            println!("hash_round shani={} avx2={}: {:>7.1} ns /key", shani, rmda, (el*1e9)/(rounds as f64*1024.0));
        }
        // full search_chunk (2^20 keys) with real pipeline
        let cfg = HashCfg { shani: true, rmd_avx2: true };
        let stop = AtomicBool::new(false);
        let counter = AtomicU64::new(0);
        let target = TARGET_HASH160;
        let ctx = SearchCtx { target: &target, cfg, gtable: &gtable, stop: &stop, counter: &counter };
        let t = Instant::now();
        let mut sc2 = Scratch::new(STRANDS);
        search_chunk(&mut sc2, RANGE_START, ROUNDS, &step, &ctx, false);
        let el = t.elapsed().as_secs_f64();
        println!("search_chunk : {:>7.1} ns /key", (el*1e9)/((1u64<<20) as f64));
    }
}
RUST_EOF

echo "Building release binary (this may take a minute)..."
cargo build --release

echo "Starting keysearch..."
exec ./target/release/keysearch "$@"

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sp1_zkvm::io;
use std::vec::Vec;

pub const PROTOCOL_VERSION: u32 = 1;
pub const STREAMING_PROTOCOL_VERSION: u32 = 2;
const FIXED_SCALE: i64 = 1 << 18;
const Q30_SCALE: i64 = 1 << 30;
const KEY_COMMIT_DOMAIN: &[u8] = b"ppmark_key_v1";
const OPENING_DOMAIN: &[u8] = b"ppmark_opening_v1";
const STREAMING_OPENING_DOMAIN: &[u8] = b"ppmark_opening_v2";
const SAMPLE_INDICES_DOMAIN: &[u8] = b"ppmark_sample_indices_v1";
const TRACE_COMMITMENT_DOMAIN: &[u8] = b"ppmark_trace_v2";
const LUT_BYTES: &[u8] = include_bytes!("../../../tables/invcdf_gaussian.bin");
const LUT_LEN: usize = LUT_BYTES.len() / 4;
const LUT_HASH: [u8; 32] = [
    0x1c, 0xfb, 0xba, 0x1e, 0x07, 0xe3, 0x43, 0x86, 0xa5, 0x76, 0x26, 0xf8, 0x2a, 0xeb, 0x43, 0xe6,
    0xed, 0x54, 0xba, 0x70, 0xa3, 0xea, 0x6b, 0x7d, 0x96, 0x4d, 0xc6, 0x6a, 0x84, 0x10, 0x74, 0xa8,
];

/// The complete public claim carried by an Attest receipt.
///
/// The host serializes exactly this struct into SP1 public values and compares
/// those bytes with the values embedded in the receipt during verification.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AttestationStatement {
    pub protocol_version: u32,
    pub ctx_hash: [u8; 32],
    pub binding: [u8; 32],
    pub producer_key_commitment: [u8; 32],
    pub image_hash: [u8; 32],
    pub sample_merkle_root: [u8; 32],
    pub sample_indices_hash: [u8; 32],
    pub challenge_seed: [u8; 32],
    pub opening_digest: [u8; 32],
    pub lut_hash: [u8; 32],
    pub opening_k: u32,
    pub alpha_effective_fixed: i64,
    pub scale_factor_q30: i64,
    pub sample_count: u32,
    pub width: u32,
    pub height: u32,
    pub rs_n: u32,
    pub rs_k: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SampleEntry {
    pub index: u32,
    pub gaussian_fixed: i64,
    pub combined_fixed: i64,
    pub bit: u8,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GuestInputs {
    pub public: AttestationStatement,
    pub secret_key: [u8; 32],
    pub codeword: Vec<u8>,
    pub samples: Vec<SampleEntry>,
}

/// Protocol-v2 statement used by the full-trace streaming-commitment
/// entrypoint. V1 remains the submitted Merkle relation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamingAttestationStatement {
    pub protocol_version: u32,
    pub ctx_hash: [u8; 32],
    pub binding: [u8; 32],
    pub producer_key_commitment: [u8; 32],
    pub image_hash: [u8; 32],
    pub trace_commitment: [u8; 32],
    pub sample_indices_hash: [u8; 32],
    pub challenge_seed: [u8; 32],
    pub opening_digest: [u8; 32],
    pub lut_hash: [u8; 32],
    pub opening_k: u32,
    pub alpha_effective_fixed: i64,
    pub scale_factor_q30: i64,
    pub sample_count: u32,
    pub width: u32,
    pub height: u32,
    pub rs_n: u32,
    pub rs_k: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingGuestInputs {
    pub public: StreamingAttestationStatement,
    pub secret_key: [u8; 32],
    pub codeword: Vec<u8>,
    pub samples: Vec<SampleEntry>,
}

fn hash_leaf(sample: &SampleEntry) -> [u8; 32] {
    let mut buf = [0u8; 4 + 8 + 8 + 1];
    buf[0..4].copy_from_slice(&sample.index.to_le_bytes());
    buf[4..12].copy_from_slice(&sample.gaussian_fixed.to_le_bytes());
    buf[12..20].copy_from_slice(&sample.combined_fixed.to_le_bytes());
    buf[20] = sample.bit;
    Sha256::digest(buf).into()
}

fn compute_merkle_root(samples: &[SampleEntry]) -> [u8; 32] {
    if samples.is_empty() {
        panic!("P2: no samples provided");
    }
    let mut level: Vec<[u8; 32]> = samples.iter().map(hash_leaf).collect();
    let target = level.len().next_power_of_two();
    level.resize(target, [0u8; 32]);
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            let mut hasher = Sha256::new();
            hasher.update(pair[0]);
            hasher.update(pair[1]);
            next.push(hasher.finalize().into());
        }
        level = next;
    }
    level[0]
}

fn digest_modulo(digest: &[u8; 32], modulus: u32) -> u32 {
    if modulus == 0 {
        panic!("zero modulus");
    }
    let mut rem: u32 = 0;
    for byte in digest {
        rem = ((rem as u64 * 256 + *byte as u64) % modulus as u64) as u32;
    }
    rem
}

fn derive_opening_positions(seed: [u8; 32], k: u32, sample_count: u32) -> Vec<u32> {
    if k == 0 || k > sample_count {
        panic!("P3: opening_k must be in [1, sample_count]");
    }
    let mut positions: Vec<u32> = Vec::with_capacity(k as usize);
    let mut counter: u32 = 0;
    while positions.len() < k as usize {
        let mut hasher = Sha256::new();
        hasher.update(seed);
        hasher.update(counter.to_le_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        let pos = digest_modulo(&digest, sample_count);
        if !positions.contains(&pos) {
            positions.push(pos);
        }
        counter = counter
            .checked_add(1)
            .expect("P3: challenge counter overflow");
    }
    positions
}

fn compute_opening_digest(samples: &[SampleEntry], positions: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for pos in positions {
        let sample = samples
            .get(*pos as usize)
            .unwrap_or_else(|| panic!("P3: opening position {pos} out of bounds"));
        hasher.update(pos.to_le_bytes());
        hasher.update(sample.index.to_le_bytes());
        hasher.update(sample.gaussian_fixed.to_le_bytes());
        hasher.update(sample.combined_fixed.to_le_bytes());
        hasher.update([sample.bit]);
    }
    hasher.finalize().into()
}

fn compute_sample_indices_hash(samples: &[SampleEntry]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SAMPLE_INDICES_DOMAIN);
    for sample in samples {
        hasher.update(sample.index.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Canonical collision-resistant commitment to the complete ordered trace.
/// The explicit domain and length prevent cross-protocol and truncation
/// ambiguity; every entry uses one fixed-width little-endian encoding.
pub fn compute_trace_commitment(samples: &[SampleEntry]) -> [u8; 32] {
    let sample_count = u32::try_from(samples.len()).expect("trace length exceeds u32");
    let mut hasher = Sha256::new();
    hasher.update(TRACE_COMMITMENT_DOMAIN);
    hasher.update(sample_count.to_le_bytes());
    for sample in samples {
        hasher.update(sample.index.to_le_bytes());
        hasher.update(sample.gaussian_fixed.to_le_bytes());
        hasher.update(sample.combined_fixed.to_le_bytes());
        hasher.update([sample.bit]);
    }
    hasher.finalize().into()
}

fn compute_challenge_seed(public: &AttestationStatement) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(OPENING_DOMAIN);
    hasher.update(public.image_hash);
    hasher.update(public.ctx_hash);
    hasher.update(public.sample_merkle_root);
    hasher.finalize().into()
}

fn compute_streaming_challenge_seed(public: &StreamingAttestationStatement) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(STREAMING_OPENING_DOMAIN);
    hasher.update(public.image_hash);
    hasher.update(public.ctx_hash);
    hasher.update(public.trace_commitment);
    hasher.finalize().into()
}

fn gf_tables() -> ([u8; 512], [u8; 256]) {
    let mut exp = [0u8; 512];
    let mut log = [0u8; 256];
    let mut x: u16 = 1;
    for i in 0..255usize {
        exp[i] = x as u8;
        log[x as usize] = i as u8;
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= 0x11d;
        }
    }
    for i in 255..512usize {
        exp[i] = exp[i - 255];
    }
    (exp, log)
}

fn gf_mul(x: u8, y: u8, exp: &[u8; 512], log: &[u8; 256]) -> u8 {
    if x == 0 || y == 0 {
        return 0;
    }
    exp[log[x as usize] as usize + log[y as usize] as usize]
}

fn poly_mul(left: &[u8], right: &[u8], exp: &[u8; 512], log: &[u8; 256]) -> Vec<u8> {
    let mut out = vec![0u8; left.len() + right.len() - 1];
    for (i, a) in left.iter().enumerate() {
        for (j, b) in right.iter().enumerate() {
            out[i + j] ^= gf_mul(*a, *b, exp, log);
        }
    }
    out
}

/// Systematic RS encoder matching reedsolo.RSCodec(nsym), with GF(2^8),
/// primitive polynomial 0x11d, generator 2, and first consecutive root 0.
fn rs_encode(message: &[u8], parity_symbols: usize) -> Vec<u8> {
    if message.is_empty() || parity_symbols == 0 || message.len() + parity_symbols > 255 {
        panic!("P4: unsupported Reed-Solomon dimensions");
    }
    let (exp, log) = gf_tables();
    let mut generator = vec![1u8];
    for i in 0..parity_symbols {
        generator = poly_mul(&generator, &[1u8, exp[i]], &exp, &log);
    }

    let mut work = vec![0u8; message.len() + parity_symbols];
    work[..message.len()].copy_from_slice(message);
    for i in 0..message.len() {
        let coefficient = work[i];
        if coefficient != 0 {
            for (j, generator_coefficient) in generator.iter().enumerate() {
                work[i + j] ^= gf_mul(*generator_coefficient, coefficient, &exp, &log);
            }
        }
    }
    work[..message.len()].copy_from_slice(message);
    work
}

fn codeword_bit(codeword: &[u8], bit_index: u32) -> u8 {
    let byte_index = (bit_index / 8) as usize;
    let shift = 7 - (bit_index % 8);
    (codeword[byte_index] >> shift) & 1
}

fn expected_bit(binding: [u8; 32], index: u32, width: u32, codeword: &[u8]) -> u8 {
    let x = index % width;
    let y = index / width;
    let mut hasher = Sha256::new();
    hasher.update(binding);
    for value in [x, y, 777u32] {
        let mut encoded = [0u8; 32];
        encoded[28..].copy_from_slice(&value.to_be_bytes());
        hasher.update(encoded);
    }
    let digest: [u8; 32] = hasher.finalize().into();
    codeword_bit(
        codeword,
        digest_modulo(&digest, (codeword.len() * 8) as u32),
    )
}

fn lut_index(binding: [u8; 32], flattened_index: u32) -> usize {
    let counter = flattened_index
        .checked_add(1)
        .expect("P4: flattened index overflow");
    let mut encoded = [0u8; 32];
    encoded[28..].copy_from_slice(&counter.to_be_bytes());
    let mut hasher = Sha256::new();
    hasher.update(binding);
    hasher.update(encoded);
    let digest: [u8; 32] = hasher.finalize().into();
    let uniform = u32::from_be_bytes(digest[28..32].try_into().expect("four-byte suffix"));
    ((uniform as u64 * (LUT_LEN as u64 - 1)) >> 32) as usize
}

/// Convert an IEEE-754 f32 bit pattern to Q18 using round-to-nearest,
/// ties-to-even. Python's round(float(np.float32(x)) * 2^18) has the same rule.
fn f32_bits_to_q18(bits: u32) -> i64 {
    let negative = bits >> 31 != 0;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = (bits & 0x7f_ffff) as u64;
    if exponent == 0xff {
        panic!("P4: non-finite LUT value");
    }
    if exponent == 0 && fraction == 0 {
        return 0;
    }
    let (mantissa, power) = if exponent == 0 {
        (fraction, -131i32)
    } else {
        ((1u64 << 23) | fraction, exponent - 132)
    };
    let magnitude = if power >= 0 {
        mantissa
            .checked_shl(power as u32)
            .expect("P4: LUT fixed-point overflow")
    } else {
        let shift = (-power) as u32;
        if shift >= 64 {
            0
        } else if shift == 0 {
            mantissa
        } else {
            let quotient = mantissa >> shift;
            let mask = (1u64 << shift) - 1;
            let remainder = mantissa & mask;
            let half = 1u64 << (shift - 1);
            quotient + u64::from(remainder > half || (remainder == half && quotient & 1 == 1))
        }
    };
    let signed = i64::try_from(magnitude).expect("P4: LUT value does not fit i64");
    if negative {
        -signed
    } else {
        signed
    }
}

fn lut_q18(index: usize) -> i64 {
    let offset = index.checked_mul(4).expect("P4: LUT offset overflow");
    let bits = u32::from_le_bytes(
        LUT_BYTES[offset..offset + 4]
            .try_into()
            .expect("four-byte LUT element"),
    );
    f32_bits_to_q18(bits)
}

fn round_div_signed(value: i128, denominator: i128) -> i64 {
    if denominator <= 0 {
        panic!("P4: denominator must be positive");
    }
    let half = denominator / 2;
    let rounded = if value >= 0 {
        (value + half) / denominator
    } else {
        -((-value + half) / denominator)
    };
    i64::try_from(rounded).expect("P4: scaled Gaussian does not fit i64")
}

fn expected_gaussian_fixed(binding: [u8; 32], flattened_index: u32, scale_factor_q30: i64) -> i64 {
    let raw = lut_q18(lut_index(binding, flattened_index));
    round_div_signed(raw as i128 * scale_factor_q30 as i128, Q30_SCALE as i128)
}

fn enforce_p4(sample: &SampleEntry, public: &AttestationStatement, codeword: &[u8]) {
    let bit = expected_bit(public.binding, sample.index, public.width, codeword);
    if sample.bit != bit {
        panic!("P4: RS-derived bit mismatch at sample {}", sample.index);
    }
    let gaussian = expected_gaussian_fixed(public.binding, sample.index, public.scale_factor_q30);
    if sample.gaussian_fixed != gaussian {
        panic!(
            "P4: inverse-CDF Gaussian mismatch at sample {}",
            sample.index
        );
    }
    let sign = if bit == 1 { 1i64 } else { -1i64 };
    let combined = gaussian
        .checked_add(
            public
                .alpha_effective_fixed
                .checked_mul(sign)
                .expect("P4: alpha/sign overflow"),
        )
        .expect("P4: combined value overflow");
    if sample.combined_fixed != combined {
        panic!("P4: embedding relation mismatch at sample {}", sample.index);
    }
}

fn enforce_streaming_p4(
    sample: &SampleEntry,
    public: &StreamingAttestationStatement,
    codeword: &[u8],
) {
    let bit = expected_bit(public.binding, sample.index, public.width, codeword);
    if sample.bit != bit {
        panic!("P4: RS-derived bit mismatch at sample {}", sample.index);
    }
    let gaussian = expected_gaussian_fixed(public.binding, sample.index, public.scale_factor_q30);
    if sample.gaussian_fixed != gaussian {
        panic!(
            "P4: inverse-CDF Gaussian mismatch at sample {}",
            sample.index
        );
    }
    let sign = if bit == 1 { 1i64 } else { -1i64 };
    let combined = gaussian
        .checked_add(
            public
                .alpha_effective_fixed
                .checked_mul(sign)
                .expect("P4: alpha/sign overflow"),
        )
        .expect("P4: combined value overflow");
    if sample.combined_fixed != combined {
        panic!("P4: embedding relation mismatch at sample {}", sample.index);
    }
}

fn validate_statement_shape(public: &AttestationStatement, samples_len: usize) {
    if public.protocol_version != PROTOCOL_VERSION {
        panic!("unsupported protocol version");
    }
    if public.rs_k != 32 || public.rs_n <= public.rs_k || public.rs_n > 255 {
        panic!("P4: unsupported RS parameters");
    }
    if public.sample_count == 0 || public.sample_count as usize != samples_len {
        panic!("P2: sample_count mismatch");
    }
    if public.width == 0 || public.height == 0 {
        panic!("invalid latent dimensions");
    }
    if public.alpha_effective_fixed <= 0
        || public.alpha_effective_fixed > 16 * FIXED_SCALE
        || public.scale_factor_q30 <= 0
        || public.scale_factor_q30 > Q30_SCALE
    {
        panic!("P4: invalid fixed-point embedding parameters");
    }
    public
        .width
        .checked_mul(public.height)
        .expect("latent dimensions overflow");
}

fn validate_streaming_statement_shape(public: &StreamingAttestationStatement, samples_len: usize) {
    if public.protocol_version != STREAMING_PROTOCOL_VERSION {
        panic!("unsupported streaming protocol version");
    }
    if public.rs_k != 32 || public.rs_n <= public.rs_k || public.rs_n > 255 {
        panic!("P4: unsupported RS parameters");
    }
    if public.sample_count == 0 || public.sample_count as usize != samples_len {
        panic!("P2: sample_count mismatch");
    }
    if public.width == 0 || public.height == 0 {
        panic!("invalid latent dimensions");
    }
    if public.alpha_effective_fixed <= 0
        || public.alpha_effective_fixed > 16 * FIXED_SCALE
        || public.scale_factor_q30 <= 0
        || public.scale_factor_q30 > Q30_SCALE
    {
        panic!("P4: invalid fixed-point embedding parameters");
    }
    public
        .width
        .checked_mul(public.height)
        .expect("latent dimensions overflow");
}

pub fn guest_main() {
    let input = io::read::<GuestInputs>();
    validate_statement_shape(&input.public, input.samples.len());

    let total_pixels = input
        .public
        .width
        .checked_mul(input.public.height)
        .expect("latent dimensions overflow");
    for sample in &input.samples {
        if sample.index >= total_pixels {
            panic!("sample index out of bounds");
        }
        if sample.bit > 1 {
            panic!("sample bit must be binary");
        }
    }

    // P1: knowledge of the binding key and membership in the verifier-selected
    // producer-key trust anchor.
    let mut binding_hasher = Sha256::new();
    binding_hasher.update(b"ctx_hash");
    binding_hasher.update(input.public.ctx_hash);
    binding_hasher.update(input.secret_key);
    let binding_expected: [u8; 32] = binding_hasher.finalize().into();
    if binding_expected != input.public.binding {
        panic!("P1: binding mismatch");
    }
    let mut key_hasher = Sha256::new();
    key_hasher.update(KEY_COMMIT_DOMAIN);
    key_hasher.update(input.secret_key);
    let key_commitment: [u8; 32] = key_hasher.finalize().into();
    if key_commitment != input.public.producer_key_commitment {
        panic!("P1: producer key commitment mismatch");
    }

    // P2: commit to the complete ordered sampled trace and bind its indices to
    // the deterministic sample set reconstructed by the public verifier.
    if compute_sample_indices_hash(&input.samples) != input.public.sample_indices_hash {
        panic!("P2: sample index set mismatch");
    }
    if compute_merkle_root(&input.samples) != input.public.sample_merkle_root {
        panic!("P2: sample Merkle root mismatch");
    }

    // The LUT is part of the public relation, not a host-selected file path.
    // The ELF commits to LUT_BYTES; comparing its audited constant hash avoids
    // hashing all 256 KiB inside every proof.
    if LUT_HASH != input.public.lut_hash {
        panic!("P4: inverse-CDF LUT hash mismatch");
    }

    // Recompute the systematic RS codeword from b; a witness-supplied payload
    // can no longer select arbitrary watermark bits.
    let parity = (input.public.rs_n - input.public.rs_k) as usize;
    let expected_codeword = rs_encode(&input.public.binding, parity);
    if input.codeword != expected_codeword {
        panic!("P4: RS codeword mismatch");
    }

    // P3: image-bound Fiat-Shamir opening challenge.
    let challenge = compute_challenge_seed(&input.public);
    if challenge != input.public.challenge_seed {
        panic!("P3: challenge seed mismatch");
    }
    let positions =
        derive_opening_positions(challenge, input.public.opening_k, input.public.sample_count);

    // P4: each challenged trace entry is determined by RS(b), the embedded
    // inverse-CDF table, and the declared normalized embedding parameters.
    for pos in &positions {
        enforce_p4(
            &input.samples[*pos as usize],
            &input.public,
            &expected_codeword,
        );
    }
    if compute_opening_digest(&input.samples, &positions) != input.public.opening_digest {
        panic!("P3: opening digest mismatch");
    }

    // A single canonical serialization prevents field-order and omission bugs.
    io::commit(&input.public);
}

/// Protocol-v2 Attest relation. It retains knowledge of the complete private
/// trace but replaces the in-guest binary Merkle reconstruction with one
/// canonical streaming SHA-256 commitment.
pub fn streaming_guest_main() {
    let input = io::read::<StreamingGuestInputs>();
    validate_streaming_statement_shape(&input.public, input.samples.len());

    let total_pixels = input
        .public
        .width
        .checked_mul(input.public.height)
        .expect("latent dimensions overflow");
    for sample in &input.samples {
        if sample.index >= total_pixels {
            panic!("sample index out of bounds");
        }
        if sample.bit > 1 {
            panic!("sample bit must be binary");
        }
    }

    let mut binding_hasher = Sha256::new();
    binding_hasher.update(b"ctx_hash");
    binding_hasher.update(input.public.ctx_hash);
    binding_hasher.update(input.secret_key);
    let binding_expected: [u8; 32] = binding_hasher.finalize().into();
    if binding_expected != input.public.binding {
        panic!("P1: binding mismatch");
    }
    let mut key_hasher = Sha256::new();
    key_hasher.update(KEY_COMMIT_DOMAIN);
    key_hasher.update(input.secret_key);
    let key_commitment: [u8; 32] = key_hasher.finalize().into();
    if key_commitment != input.public.producer_key_commitment {
        panic!("P1: producer key commitment mismatch");
    }

    if compute_sample_indices_hash(&input.samples) != input.public.sample_indices_hash {
        panic!("P2: sample index set mismatch");
    }
    if compute_trace_commitment(&input.samples) != input.public.trace_commitment {
        panic!("P2: trace commitment mismatch");
    }

    if LUT_HASH != input.public.lut_hash {
        panic!("P4: inverse-CDF LUT hash mismatch");
    }
    let parity = (input.public.rs_n - input.public.rs_k) as usize;
    let expected_codeword = rs_encode(&input.public.binding, parity);
    if input.codeword != expected_codeword {
        panic!("P4: RS codeword mismatch");
    }

    let challenge = compute_streaming_challenge_seed(&input.public);
    if challenge != input.public.challenge_seed {
        panic!("P3: challenge seed mismatch");
    }
    let positions =
        derive_opening_positions(challenge, input.public.opening_k, input.public.sample_count);
    for pos in &positions {
        enforce_streaming_p4(
            &input.samples[*pos as usize],
            &input.public,
            &expected_codeword,
        );
    }
    if compute_opening_digest(&input.samples, &positions) != input.public.opening_digest {
        panic!("P3: opening digest mismatch");
    }

    io::commit(&input.public);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reed_solomon_matches_reedsolo_vector() {
        let message: Vec<u8> = (0u8..32u8).collect();
        let expected = hex::decode(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f\
             f57095816f2872728e21354bc5470fa6e675cbed022673091964a0bde3fb0bb4"
                .replace(' ', ""),
        )
        .unwrap();
        assert_eq!(rs_encode(&message, 32), expected);
    }

    #[test]
    fn lut_and_fixed_relation_match_python_vector() {
        assert_eq!(LUT_LEN, 65_536);
        let lut_hash: [u8; 32] = Sha256::digest(LUT_BYTES).into();
        assert_eq!(lut_hash, LUT_HASH);
        assert_eq!(
            hex::encode(lut_hash),
            "1cfbba1e07e34386a57626f82aeb43e6ed54ba70a3ea6b7d964dc66a841074a8"
        );
        let mut binding = [0u8; 32];
        for (i, byte) in binding.iter_mut().enumerate() {
            *byte = i as u8;
        }
        assert_eq!(lut_index(binding, 0), 20_011);
        assert_eq!(expected_gaussian_fixed(binding, 17, 260_420_644), -24_470);
    }

    #[test]
    fn opening_positions_are_unique_and_deterministic() {
        let seed = [7u8; 32];
        let left = derive_opening_positions(seed, 32, 1_000);
        let right = derive_opening_positions(seed, 32, 1_000);
        assert_eq!(left, right);
        let mut sorted = left.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 32);
    }

    #[test]
    fn streaming_trace_commitment_binds_length_and_order() {
        let samples = vec![
            SampleEntry {
                index: 7,
                gaussian_fixed: -11,
                combined_fixed: 13,
                bit: 1,
            },
            SampleEntry {
                index: 9,
                gaussian_fixed: 17,
                combined_fixed: -19,
                bit: 0,
            },
        ];
        let digest = compute_trace_commitment(&samples);
        assert_eq!(
            hex::encode(digest),
            "3f2618cf432fef251e643d44b0585bf02edfb7bd25d542e78a40cb9ad28013ae"
        );
        let mut reversed = samples.clone();
        reversed.reverse();
        assert_ne!(digest, compute_trace_commitment(&reversed));
        assert_ne!(digest, compute_trace_commitment(&samples[..1]));
    }
}

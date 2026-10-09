//! Model-free SP1 execution-cost comparison for the current and pre-P1--P4 guests.
//!
//! The legacy mode deliberately exists only in this non-default diagnostic
//! example. Production verification refuses legacy SP1 statements.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sp1_genguard_guest::{
    AttestationStatement, GuestInputs, SampleEntry, StreamingAttestationStatement,
    StreamingGuestInputs,
};
use sp1_sdk::{
    blocking::{Prover, ProverClient},
    Elf, SP1PublicValues, SP1Stdin,
};
use std::{env, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStatement {
    protocol_version: u32,
    ctx_hash: String,
    binding: String,
    producer_key_commitment: String,
    image_hash: String,
    sample_merkle_root: String,
    sample_indices_hash: String,
    challenge_seed: String,
    opening_digest: String,
    lut_hash: String,
    opening_k: u32,
    alpha_effective_fixed: String,
    scale_factor_q30: String,
    sample_count: u32,
    width: u32,
    height: u32,
    rs_n: u32,
    rs_k: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStreamingStatement {
    protocol_version: u32,
    ctx_hash: String,
    binding: String,
    producer_key_commitment: String,
    image_hash: String,
    trace_commitment: String,
    sample_indices_hash: String,
    challenge_seed: String,
    opening_digest: String,
    lut_hash: String,
    opening_k: u32,
    alpha_effective_fixed: String,
    scale_factor_q30: String,
    sample_count: u32,
    width: u32,
    height: u32,
    rs_n: u32,
    rs_k: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSample {
    index: u32,
    gaussian_fixed: String,
    combined_fixed: String,
    bit: u8,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWitness {
    secret_key: String,
    codeword: String,
    samples: Vec<RawSample>,
}

// These four structs reproduce the exact bincode/serde shape of the guest at
// Git HEAD before the P1--P4 alignment pass.
#[derive(Debug, Serialize, Deserialize)]
struct LegacyPublicInputs {
    ctx_hash: [u8; 32],
    binding: [u8; 32],
    sample_merkle_root: [u8; 32],
    alpha: i64,
    sample_merkle_depth: u32,
    model_id: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    sample_count: Option<u32>,
    lut_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LegacySampleEntry {
    index: u32,
    gaussian_fixed: i64,
    combined_fixed: i64,
    bit: u8,
    merkle_path: Vec<[u8; 32]>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LegacyGuestInputs {
    public: LegacyPublicInputs,
    seed: [u8; 32],
    secret_key: [u8; 32],
    codeword: Vec<u8>,
    samples: Vec<LegacySampleEntry>,
}

fn parse_hex32(raw: &str, name: &str) -> Result<[u8; 32]> {
    let text = raw
        .trim()
        .strip_prefix("0x")
        .or_else(|| raw.trim().strip_prefix("0X"))
        .unwrap_or(raw.trim());
    if text.len() != 64 {
        bail!("{name} must contain exactly 64 hex digits");
    }
    hex::decode(text)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("{name} must be 32 bytes"))
}

fn parse_i64(raw: &str, name: &str) -> Result<i64> {
    raw.parse::<i64>()
        .with_context(|| format!("{name} must be a decimal i64"))
}

fn load_inputs(
    public_path: &PathBuf,
    witness_path: &PathBuf,
) -> Result<(GuestInputs, RawStatement)> {
    let raw_public: RawStatement = serde_json::from_str(&fs::read_to_string(public_path)?)?;
    let raw_witness: RawWitness = serde_json::from_str(&fs::read_to_string(witness_path)?)?;
    let public = AttestationStatement {
        protocol_version: raw_public.protocol_version,
        ctx_hash: parse_hex32(&raw_public.ctx_hash, "ctx_hash")?,
        binding: parse_hex32(&raw_public.binding, "binding")?,
        producer_key_commitment: parse_hex32(
            &raw_public.producer_key_commitment,
            "producer_key_commitment",
        )?,
        image_hash: parse_hex32(&raw_public.image_hash, "image_hash")?,
        sample_merkle_root: parse_hex32(&raw_public.sample_merkle_root, "sample_merkle_root")?,
        sample_indices_hash: parse_hex32(&raw_public.sample_indices_hash, "sample_indices_hash")?,
        challenge_seed: parse_hex32(&raw_public.challenge_seed, "challenge_seed")?,
        opening_digest: parse_hex32(&raw_public.opening_digest, "opening_digest")?,
        lut_hash: parse_hex32(&raw_public.lut_hash, "lut_hash")?,
        opening_k: raw_public.opening_k,
        alpha_effective_fixed: parse_i64(
            &raw_public.alpha_effective_fixed,
            "alpha_effective_fixed",
        )?,
        scale_factor_q30: parse_i64(&raw_public.scale_factor_q30, "scale_factor_q30")?,
        sample_count: raw_public.sample_count,
        width: raw_public.width,
        height: raw_public.height,
        rs_n: raw_public.rs_n,
        rs_k: raw_public.rs_k,
    };
    let secret_key = parse_hex32(&raw_witness.secret_key, "secret_key")?;
    let codeword = hex::decode(
        raw_witness
            .codeword
            .trim()
            .strip_prefix("0x")
            .or_else(|| raw_witness.codeword.trim().strip_prefix("0X"))
            .unwrap_or(raw_witness.codeword.trim()),
    )?;
    let samples = raw_witness
        .samples
        .into_iter()
        .map(|sample| {
            Ok(SampleEntry {
                index: sample.index,
                gaussian_fixed: parse_i64(&sample.gaussian_fixed, "gaussian_fixed")?,
                combined_fixed: parse_i64(&sample.combined_fixed, "combined_fixed")?,
                bit: sample.bit,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if samples.len() != public.sample_count as usize {
        bail!("sample count differs between statement and witness");
    }
    Ok((
        GuestInputs {
            public,
            secret_key,
            codeword,
            samples,
        },
        raw_public,
    ))
}

fn load_streaming_inputs(
    public_path: &PathBuf,
    witness_path: &PathBuf,
) -> Result<StreamingGuestInputs> {
    let raw_public: RawStreamingStatement =
        serde_json::from_str(&fs::read_to_string(public_path)?)?;
    let raw_witness: RawWitness = serde_json::from_str(&fs::read_to_string(witness_path)?)?;
    let public = StreamingAttestationStatement {
        protocol_version: raw_public.protocol_version,
        ctx_hash: parse_hex32(&raw_public.ctx_hash, "ctx_hash")?,
        binding: parse_hex32(&raw_public.binding, "binding")?,
        producer_key_commitment: parse_hex32(
            &raw_public.producer_key_commitment,
            "producer_key_commitment",
        )?,
        image_hash: parse_hex32(&raw_public.image_hash, "image_hash")?,
        trace_commitment: parse_hex32(&raw_public.trace_commitment, "trace_commitment")?,
        sample_indices_hash: parse_hex32(&raw_public.sample_indices_hash, "sample_indices_hash")?,
        challenge_seed: parse_hex32(&raw_public.challenge_seed, "challenge_seed")?,
        opening_digest: parse_hex32(&raw_public.opening_digest, "opening_digest")?,
        lut_hash: parse_hex32(&raw_public.lut_hash, "lut_hash")?,
        opening_k: raw_public.opening_k,
        alpha_effective_fixed: parse_i64(
            &raw_public.alpha_effective_fixed,
            "alpha_effective_fixed",
        )?,
        scale_factor_q30: parse_i64(&raw_public.scale_factor_q30, "scale_factor_q30")?,
        sample_count: raw_public.sample_count,
        width: raw_public.width,
        height: raw_public.height,
        rs_n: raw_public.rs_n,
        rs_k: raw_public.rs_k,
    };
    let secret_key = parse_hex32(&raw_witness.secret_key, "secret_key")?;
    let codeword = hex::decode(
        raw_witness
            .codeword
            .trim()
            .strip_prefix("0x")
            .or_else(|| raw_witness.codeword.trim().strip_prefix("0X"))
            .unwrap_or(raw_witness.codeword.trim()),
    )?;
    let samples = raw_witness
        .samples
        .into_iter()
        .map(|sample| {
            Ok(SampleEntry {
                index: sample.index,
                gaussian_fixed: parse_i64(&sample.gaussian_fixed, "gaussian_fixed")?,
                combined_fixed: parse_i64(&sample.combined_fixed, "combined_fixed")?,
                bit: sample.bit,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if samples.len() != public.sample_count as usize {
        bail!("sample count differs between streaming statement and witness");
    }
    Ok(StreamingGuestInputs {
        public,
        secret_key,
        codeword,
        samples,
    })
}

fn hash_leaf(sample: &SampleEntry) -> [u8; 32] {
    let mut bytes = [0u8; 21];
    bytes[0..4].copy_from_slice(&sample.index.to_le_bytes());
    bytes[4..12].copy_from_slice(&sample.gaussian_fixed.to_le_bytes());
    bytes[12..20].copy_from_slice(&sample.combined_fixed.to_le_bytes());
    bytes[20] = sample.bit;
    Sha256::digest(bytes).into()
}

fn merkle_root_and_paths(samples: &[SampleEntry]) -> ([u8; 32], Vec<Vec<[u8; 32]>>) {
    let real_len = samples.len();
    let target = real_len.next_power_of_two();
    let mut level: Vec<[u8; 32]> = samples.iter().map(hash_leaf).collect();
    level.resize(target, [0u8; 32]);
    let mut positions: Vec<usize> = (0..real_len).collect();
    let mut paths = vec![Vec::new(); real_len];
    while level.len() > 1 {
        for (sample_position, node_position) in positions.iter().copied().enumerate() {
            paths[sample_position].push(level[node_position ^ 1]);
        }
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            let mut hasher = Sha256::new();
            hasher.update(pair[0]);
            hasher.update(pair[1]);
            next.push(hasher.finalize().into());
        }
        for position in &mut positions {
            *position /= 2;
        }
        level = next;
    }
    (level[0], paths)
}

fn legacy_inputs(current: &GuestInputs, raw: &RawStatement) -> Result<LegacyGuestInputs> {
    let (root, paths) = merkle_root_and_paths(&current.samples);
    if root != current.public.sample_merkle_root {
        bail!("derived legacy Merkle root differs from current statement");
    }
    let depth = paths.first().map_or(0, |path| path.len() as u32);
    let samples = current
        .samples
        .iter()
        .zip(paths)
        .map(|(sample, merkle_path)| LegacySampleEntry {
            index: sample.index,
            gaussian_fixed: sample.gaussian_fixed,
            combined_fixed: sample.combined_fixed,
            bit: sample.bit,
            merkle_path,
        })
        .collect();
    Ok(LegacyGuestInputs {
        public: LegacyPublicInputs {
            ctx_hash: current.public.ctx_hash,
            binding: current.public.binding,
            sample_merkle_root: current.public.sample_merkle_root,
            alpha: current.public.alpha_effective_fixed,
            sample_merkle_depth: depth,
            model_id: Some("cycle-comparison".to_owned()),
            width: Some(raw.width),
            height: Some(raw.height),
            sample_count: Some(raw.sample_count),
            lut_path: None,
        },
        seed: [0u8; 32],
        secret_key: current.secret_key,
        codeword: current.codeword.clone(),
        samples,
    })
}

fn argument(args: &[String], flag: &str) -> Result<PathBuf> {
    let index = args
        .iter()
        .position(|arg| arg == flag)
        .with_context(|| format!("{flag} is required"))?;
    Ok(PathBuf::from(
        args.get(index + 1)
            .with_context(|| format!("{flag} value is required"))?,
    ))
}

pub(crate) struct PreparedRun {
    pub(crate) stdin: SP1Stdin,
    pub(crate) sample_count: usize,
    pub(crate) opening_k: u32,
    pub(crate) expected_public_values: SP1PublicValues,
}

pub(crate) fn prepare_run(
    mode: &str,
    public_path: &PathBuf,
    witness_path: &PathBuf,
) -> Result<PreparedRun> {
    let mut stdin = SP1Stdin::new();
    let (sample_count, opening_k, expected_public_values) = if mode == "streaming" {
        let streaming = load_streaming_inputs(public_path, witness_path)?;
        let count = streaming.samples.len();
        let opening_k = streaming.public.opening_k;
        let mut expected = SP1PublicValues::new();
        expected.write(&streaming.public);
        stdin.write(&streaming);
        (count, opening_k, expected)
    } else {
        let (current, raw) = load_inputs(public_path, witness_path)?;
        let count = current.samples.len();
        let opening_k = current.public.opening_k;
        let mut expected = SP1PublicValues::new();
        if mode == "current" {
            expected.write(&current.public);
            stdin.write(&current);
        } else if mode == "legacy" {
            // The pre-alignment guest committed these values separately, in
            // this order, rather than committing the complete statement.
            expected.write(&current.public.sample_merkle_root);
            expected.write(&current.public.binding);
            stdin.write(&legacy_inputs(&current, &raw)?);
        } else {
            bail!("mode must be current, legacy, or streaming");
        }
        (count, opening_k, expected)
    };
    Ok(PreparedRun {
        stdin,
        sample_count,
        opening_k,
        expected_public_values,
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("");
    if !matches!(mode, "current" | "legacy" | "streaming") {
        bail!(
            "usage: cycle_compare current|legacy|streaming --elf FILE --public FILE --witness FILE"
        );
    }
    let elf = Elf::from(fs::read(argument(&args, "--elf")?)?);
    let public_path = argument(&args, "--public")?;
    let witness_path = argument(&args, "--witness")?;
    let prepared = prepare_run(mode, &public_path, &witness_path)?;

    sp1_sdk::setup_logger();
    let (_public_values, report) = ProverClient::from_env()
        .execute(elf, prepared.stdin)
        .run()?;
    println!(
        "mode={mode} samples={} instructions={} syscalls={} touched_memory={} gas={}",
        prepared.sample_count,
        report.total_instruction_count(),
        report.total_syscall_count(),
        report.touched_memory_addresses,
        report
            .gas()
            .map_or_else(|| "unavailable".to_owned(), |gas| gas.to_string())
    );
    if matches!(env::var("PPMARK_CYCLE_DETAIL").as_deref(), Ok("true")) {
        println!("{report}");
    }
    Ok(())
}

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use sp1_genguard_guest::{
    SampleEntry, StreamingAttestationStatement, StreamingGuestInputs, STREAMING_PROTOCOL_VERSION,
};
use sp1_sdk::{
    blocking::{EnvProver, EnvProvingKey, ProveRequest, Prover, ProverClient},
    include_elf, Elf, ProvingKey, SP1ProofMode, SP1ProofWithPublicValues, SP1PublicValues,
    SP1Stdin, SP1VerifyingKey,
};
use std::{env, fs, path::PathBuf};

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

impl TryFrom<RawStreamingStatement> for StreamingAttestationStatement {
    type Error = anyhow::Error;

    fn try_from(raw: RawStreamingStatement) -> Result<Self> {
        if raw.protocol_version != STREAMING_PROTOCOL_VERSION {
            bail!(
                "production host requires protocol version {STREAMING_PROTOCOL_VERSION}; received {}",
                raw.protocol_version
            );
        }
        Ok(Self {
            protocol_version: raw.protocol_version,
            ctx_hash: parse_hex_32(&raw.ctx_hash, "ctx_hash")?,
            binding: parse_hex_32(&raw.binding, "binding")?,
            producer_key_commitment: parse_hex_32(
                &raw.producer_key_commitment,
                "producer_key_commitment",
            )?,
            image_hash: parse_hex_32(&raw.image_hash, "image_hash")?,
            trace_commitment: parse_hex_32(&raw.trace_commitment, "trace_commitment")?,
            sample_indices_hash: parse_hex_32(&raw.sample_indices_hash, "sample_indices_hash")?,
            challenge_seed: parse_hex_32(&raw.challenge_seed, "challenge_seed")?,
            opening_digest: parse_hex_32(&raw.opening_digest, "opening_digest")?,
            lut_hash: parse_hex_32(&raw.lut_hash, "lut_hash")?,
            opening_k: raw.opening_k,
            alpha_effective_fixed: parse_i64(&raw.alpha_effective_fixed, "alpha_effective_fixed")?,
            scale_factor_q30: parse_i64(&raw.scale_factor_q30, "scale_factor_q30")?,
            sample_count: raw.sample_count,
            width: raw.width,
            height: raw.height,
            rs_n: raw.rs_n,
            rs_k: raw.rs_k,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSampleEntry {
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
    samples: Vec<RawSampleEntry>,
}

fn parse_hex_32(raw: &str, name: &str) -> Result<[u8; 32]> {
    let text = raw
        .trim()
        .strip_prefix("0x")
        .or_else(|| raw.trim().strip_prefix("0X"))
        .unwrap_or(raw.trim());
    if text.len() != 64 {
        bail!("{name} must contain exactly 64 hex digits");
    }
    let bytes = hex::decode(text).with_context(|| format!("invalid hex in {name}"))?;
    bytes
        .try_into()
        .map_err(|_| anyhow!("{name} must decode to 32 bytes"))
}

fn parse_i64(raw: &str, name: &str) -> Result<i64> {
    raw.parse::<i64>()
        .with_context(|| format!("{name} must be a signed decimal i64"))
}

fn parse_codeword(raw: &str) -> Result<Vec<u8>> {
    let text = raw
        .trim()
        .strip_prefix("0x")
        .or_else(|| raw.trim().strip_prefix("0X"))
        .unwrap_or(raw.trim());
    hex::decode(text).context("invalid codeword hex")
}

fn load_statement(path: &PathBuf) -> Result<StreamingAttestationStatement> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read public statement {}", path.display()))?;
    let raw: RawStreamingStatement = serde_json::from_str(&data)
        .with_context(|| format!("invalid protocol-v2 statement JSON {}", path.display()))?;
    raw.try_into()
}

fn parse_witness(
    path: &PathBuf,
    public: StreamingAttestationStatement,
) -> Result<StreamingGuestInputs> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read private witness {}", path.display()))?;
    let raw: RawWitness = serde_json::from_str(&data)
        .with_context(|| format!("invalid private witness JSON {}", path.display()))?;
    let secret_key = parse_hex_32(&raw.secret_key, "secret_key")?;
    let codeword = parse_codeword(&raw.codeword)?;
    if raw.samples.is_empty() {
        bail!("private witness samples must not be empty");
    }
    let samples = raw
        .samples
        .into_iter()
        .map(|sample| {
            if sample.bit > 1 {
                bail!("sample bit must be binary");
            }
            Ok(SampleEntry {
                index: sample.index,
                gaussian_fixed: parse_i64(&sample.gaussian_fixed, "gaussian_fixed")?,
                combined_fixed: parse_i64(&sample.combined_fixed, "combined_fixed")?,
                bit: sample.bit,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if samples.len() != public.sample_count as usize {
        bail!(
            "sample count differs between statement ({}) and witness ({})",
            public.sample_count,
            samples.len()
        );
    }
    Ok(StreamingGuestInputs {
        public,
        secret_key,
        codeword,
        samples,
    })
}

fn expected_public_values(statement: &StreamingAttestationStatement) -> SP1PublicValues {
    let mut expected = SP1PublicValues::new();
    expected.write(statement);
    expected
}

fn ensure_public_values_match(
    proof: &SP1ProofWithPublicValues,
    statement: &StreamingAttestationStatement,
) -> Result<()> {
    let expected = expected_public_values(statement);
    if proof.public_values.as_slice() != expected.as_slice() {
        bail!("receipt public values do not match the exact protocol-v2 statement");
    }
    Ok(())
}

fn argument_path(args: &[String], flag: &str) -> Result<PathBuf> {
    let index = args
        .iter()
        .position(|item| item == flag)
        .ok_or_else(|| anyhow!("{flag} path required"))?;
    Ok(PathBuf::from(
        args.get(index + 1)
            .ok_or_else(|| anyhow!("{flag} value missing"))?,
    ))
}

fn proof_mode_from_env() -> Result<SP1ProofMode> {
    match env::var("SP1_PROOF_MODE")
        .unwrap_or_else(|_| "core".to_owned())
        .to_ascii_lowercase()
        .as_str()
    {
        "core" => Ok(SP1ProofMode::Core),
        "compressed" => Ok(SP1ProofMode::Compressed),
        "plonk" => Ok(SP1ProofMode::Plonk),
        "groth16" => Ok(SP1ProofMode::Groth16),
        other => bail!("unsupported SP1_PROOF_MODE '{other}'"),
    }
}

fn prove(
    client: &EnvProver,
    pk: &EnvProvingKey,
    stdin: &SP1Stdin,
    mode: SP1ProofMode,
) -> Result<SP1ProofWithPublicValues> {
    let request = client.prove(pk, stdin.clone());
    Ok(match mode {
        SP1ProofMode::Core => request.core().run()?,
        SP1ProofMode::Compressed => request.compressed().run()?,
        SP1ProofMode::Plonk => request.plonk().run()?,
        SP1ProofMode::Groth16 => request.groth16().run()?,
    })
}

fn load_verifying_key(client: &EnvProver, elf: &Elf) -> Result<SP1VerifyingKey> {
    if let Ok(path) = env::var("SP1_VK_PATH") {
        return bincode::deserialize(
            &fs::read(&path).with_context(|| format!("failed to read VK {path}"))?,
        )
        .with_context(|| format!("failed to decode SP1 v6 verifying key {path}"));
    }
    let pk = client.setup(elf.clone())?;
    Ok(pk.verifying_key().clone())
}

fn validate_configured_verifying_key(pk: &EnvProvingKey) -> Result<()> {
    let Ok(path) = env::var("SP1_VK_PATH") else {
        return Ok(());
    };
    let configured: SP1VerifyingKey = bincode::deserialize(
        &fs::read(&path).with_context(|| format!("failed to read VK {path}"))?,
    )
    .with_context(|| format!("failed to decode SP1 v6 verifying key {path}"))?;
    if bincode::serialize(&configured)? != bincode::serialize(pk.verifying_key())? {
        bail!("SP1_VK_PATH does not match the protocol-v2 streaming guest");
    }
    Ok(())
}

fn reject_legacy_pk_override() -> Result<()> {
    if env::var_os("SP1_PK_PATH").is_some() {
        bail!(
            "SP1_PK_PATH is a v5-only cache format; SP1 v6.3.1 setup keys are backend-specific. \
             Reuse one long-lived host process for amortized setup instead."
        );
    }
    Ok(())
}

fn env_is_true(name: &str) -> bool {
    env::var(name)
        .map(|value| value.eq_ignore_ascii_case("true") || value == "1")
        .unwrap_or(false)
}

fn guest_elf() -> Elf {
    include_elf!("sp1-genguard-streaming")
}

fn elf_digest() -> String {
    let elf = guest_elf();
    hex::encode(Sha256::digest(&*elf))
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    sp1_sdk::setup_logger();

    match args.get(1).map(String::as_str) {
        Some("execute") => {
            let public_path = argument_path(&args, "--public")?;
            let witness_path = argument_path(&args, "--witness")?;
            let statement = load_statement(&public_path)?;
            let guest_inputs = parse_witness(&witness_path, statement)?;
            let mut stdin = SP1Stdin::new();
            stdin.write(&guest_inputs);
            let client = ProverClient::from_env();
            let (actual, report) = client.execute(guest_elf(), stdin).run()?;
            if report.exit_code != 0 {
                bail!(
                    "guest constraint execution failed (exit code {})",
                    report.exit_code
                );
            }
            let expected = expected_public_values(&guest_inputs.public);
            if actual.as_slice() != expected.as_slice() {
                bail!("guest execution emitted an unexpected public statement");
            }
            println!("SP1 Attest protocol-v2 guest execution succeeded");
        }
        Some("prove") => {
            reject_legacy_pk_override()?;
            let public_path = argument_path(&args, "--public")?;
            let witness_path = argument_path(&args, "--witness")?;
            let receipt_path = argument_path(&args, "--receipt")?;
            let statement = load_statement(&public_path)?;
            let expected_statement = statement.clone();
            let guest_inputs = parse_witness(&witness_path, statement)?;

            let mut stdin = SP1Stdin::new();
            stdin.write(&guest_inputs);
            let client = ProverClient::from_env();
            let elf = guest_elf();
            println!(
                "Running SP1 v6.3.1 setup for protocol-v2 guest (ELF digest={})...",
                elf_digest()
            );
            let pk = client.setup(elf.clone())?;
            validate_configured_verifying_key(&pk)?;

            if !env_is_true("SP1_SKIP_EXECUTE") {
                let (actual, report) = client.execute(elf, stdin.clone()).run()?;
                if report.exit_code != 0 {
                    bail!(
                        "preflight constraint execution failed (exit code {})",
                        report.exit_code
                    );
                }
                let expected = expected_public_values(&expected_statement);
                if actual.as_slice() != expected.as_slice() {
                    bail!("preflight execution emitted an unexpected public statement");
                }
            }
            let mode = proof_mode_from_env()?;
            println!("Generating protocol-v2 proof (mode={mode:?})...");
            let proof = prove(&client, &pk, &stdin, mode)?;
            if !env_is_true("SP1_SKIP_VERIFY") {
                client.verify(&proof, pk.verifying_key(), None)?;
            }
            ensure_public_values_match(&proof, &expected_statement)?;

            if let Some(parent) = receipt_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&receipt_path, bincode::serialize(&proof)?)?;
            println!(
                "SP1 Attest protocol-v2 receipt generated and statement-matched -> {}",
                receipt_path.display()
            );
        }
        Some("verify") => {
            let public_path = argument_path(&args, "--public")?;
            let receipt_path = argument_path(&args, "--receipt")?;
            let statement = load_statement(&public_path)?;
            let proof: SP1ProofWithPublicValues =
                bincode::deserialize(&fs::read(&receipt_path).with_context(|| {
                    format!("failed to read receipt {}", receipt_path.display())
                })?)?;
            let client = ProverClient::from_env();
            let vk = load_verifying_key(&client, &guest_elf())?;
            client.verify(&proof, &vk, None)?;
            ensure_public_values_match(&proof, &statement)?;
            println!(
                "SP1 Attest protocol-v2 receipt and public statement verified: {}",
                receipt_path.display()
            );
        }
        _ => {
            eprintln!(
                "Usage:\n  sp1-genguard-host execute --public FILE --witness FILE\n  \
                 sp1-genguard-host prove --public FILE --witness FILE --receipt FILE\n  \
                 sp1-genguard-host verify --public FILE --receipt FILE"
            );
            std::process::exit(2);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn statement() -> StreamingAttestationStatement {
        StreamingAttestationStatement {
            protocol_version: STREAMING_PROTOCOL_VERSION,
            ctx_hash: [1; 32],
            binding: [2; 32],
            producer_key_commitment: [3; 32],
            image_hash: [4; 32],
            trace_commitment: [5; 32],
            sample_indices_hash: [6; 32],
            challenge_seed: [7; 32],
            opening_digest: [8; 32],
            lut_hash: [9; 32],
            opening_k: 1,
            alpha_effective_fixed: 10,
            scale_factor_q30: 11,
            sample_count: 1,
            width: 1,
            height: 1,
            rs_n: 64,
            rs_k: 32,
        }
    }

    #[test]
    fn canonical_public_values_change_when_any_statement_field_changes() {
        let original = statement();
        let mut changed = original.clone();
        changed.trace_commitment[0] ^= 1;
        assert_ne!(
            expected_public_values(&original).as_slice(),
            expected_public_values(&changed).as_slice()
        );
    }

    #[test]
    fn strict_hex_parser_rejects_short_values() {
        assert!(parse_hex_32("0x01", "test").is_err());
        assert_eq!(
            parse_hex_32(&format!("0x{}", "01".repeat(32)), "test").unwrap(),
            [1; 32]
        );
    }
}

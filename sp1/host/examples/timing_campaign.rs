//! Reproducible cold/warm timing harness for the five-condition PP-Mark campaign.
//!
//! This is a diagnostic example. It does not alter the production verifier or
//! the paper relation. One process owns one prover client and one PK/VK pair so
//! warm runs measure persistent-client behavior without pooling cold and warm
//! observations.

#[path = "cycle_compare.rs"]
mod fixture;

use anyhow::{bail, Context, Result};
use fixture::prepare_run;
use serde::Serialize;
use sha2::{Digest, Sha256};
use sp1_sdk::{
    blocking::{EnvProver, EnvProvingKey, ProveRequest, Prover, ProverClient},
    Elf, ExecutionReport, ProvingKey, SP1ProofWithPublicValues, SP1PublicValues, SP1Stdin,
};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Serialize)]
struct RunMeasurement {
    measured_run: usize,
    execute_ms: f64,
    prove_ms: f64,
    verify_ms: f64,
    serialize_ms: f64,
    proof_bytes_bincode: usize,
    instructions: u64,
    syscalls: u64,
    touched_memory_addresses: u64,
    gas: Option<u64>,
}

#[derive(Debug, Serialize)]
struct Distribution {
    median: f64,
    p95: f64,
    minimum: f64,
    maximum: f64,
}

#[derive(Debug, Serialize)]
struct CampaignOutput {
    schema: &'static str,
    started_unix_ms: u128,
    relation: String,
    sample_count: usize,
    opening_k: u32,
    sp1_version: &'static str,
    proof_mode: &'static str,
    prover_backend: String,
    cuda_endpoint_configured: bool,
    elf_path: String,
    elf_bytes: usize,
    elf_sha256: String,
    warmup_runs_excluded: usize,
    measured_runs: usize,
    client_initialization_ms: f64,
    setup_ms: f64,
    verifying_key_bytes_bincode: usize,
    system: SystemMetadata,
    runs: Vec<RunMeasurement>,
    execute_ms: Distribution,
    prove_ms: Distribution,
    verify_ms: Distribution,
    serialize_ms: Distribution,
    proof_bytes_bincode: Distribution,
}

#[derive(Debug, Serialize)]
struct SystemMetadata {
    os: String,
    architecture: String,
    nvidia_smi: Option<String>,
    cuda_visible_devices: Option<String>,
    rustc: Option<String>,
}

fn argument(args: &[String], flag: &str) -> Result<String> {
    let index = args
        .iter()
        .position(|item| item == flag)
        .with_context(|| format!("{flag} is required"))?;
    args.get(index + 1)
        .cloned()
        .with_context(|| format!("{flag} value is required"))
}

fn optional_argument(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|item| item == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn usize_argument(args: &[String], flag: &str, default: usize) -> Result<usize> {
    match optional_argument(args, flag) {
        Some(raw) => raw
            .parse::<usize>()
            .with_context(|| format!("{flag} must be a non-negative integer")),
        None => Ok(default),
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn check_public_values(actual: &SP1PublicValues, expected: &SP1PublicValues) -> Result<()> {
    if actual.as_slice() != expected.as_slice() {
        bail!("public values mismatch");
    }
    Ok(())
}

fn run_once(
    client: &EnvProver,
    elf: &Elf,
    pk: &EnvProvingKey,
    stdin: &SP1Stdin,
    expected: &SP1PublicValues,
    measured_run: usize,
) -> Result<RunMeasurement> {
    let started = Instant::now();
    let (execution_public, report): (SP1PublicValues, ExecutionReport) =
        client.execute(elf.clone(), stdin.clone()).run()?;
    let execute_ms = elapsed_ms(started);
    if report.exit_code != 0 {
        bail!("guest execution failed with exit code {}", report.exit_code);
    }
    check_public_values(&execution_public, expected)?;

    let started = Instant::now();
    let proof: SP1ProofWithPublicValues = client.prove(pk, stdin.clone()).core().run()?;
    let prove_ms = elapsed_ms(started);
    check_public_values(&proof.public_values, expected)?;

    let started = Instant::now();
    client.verify(&proof, pk.verifying_key(), None)?;
    let verify_ms = elapsed_ms(started);

    let started = Instant::now();
    let encoded = bincode::serialize(&proof)?;
    let serialize_ms = elapsed_ms(started);

    Ok(RunMeasurement {
        measured_run,
        execute_ms,
        prove_ms,
        verify_ms,
        serialize_ms,
        proof_bytes_bincode: encoded.len(),
        instructions: report.total_instruction_count(),
        syscalls: report.total_syscall_count(),
        touched_memory_addresses: report.touched_memory_addresses,
        gas: report.gas(),
    })
}

fn distribution(mut values: Vec<f64>) -> Distribution {
    values.sort_by(f64::total_cmp);
    let count = values.len();
    let median = if count % 2 == 0 {
        (values[count / 2 - 1] + values[count / 2]) / 2.0
    } else {
        values[count / 2]
    };
    let p95_index = ((count as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(count - 1);
    Distribution {
        median,
        p95: values[p95_index],
        minimum: values[0],
        maximum: values[count - 1],
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn system_metadata() -> SystemMetadata {
    SystemMetadata {
        os: env::consts::OS.to_owned(),
        architecture: env::consts::ARCH.to_owned(),
        nvidia_smi: command_output(
            "nvidia-smi",
            &[
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
        ),
        cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES").ok(),
        rustc: command_output("rustc", &["--version"]),
    }
}

fn write_output(path: &Path, output: &CampaignOutput) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(output)?)
        .with_context(|| format!("write campaign output {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let relation = args.get(1).map(String::as_str).unwrap_or("");
    if !matches!(relation, "legacy" | "current" | "streaming") {
        bail!(
            "usage: timing_campaign legacy|current|streaming --elf FILE --public FILE \
             --witness FILE [--warmup N] [--runs N] [--output FILE]"
        );
    }
    let elf_path = PathBuf::from(argument(&args, "--elf")?);
    let public_path = PathBuf::from(argument(&args, "--public")?);
    let witness_path = PathBuf::from(argument(&args, "--witness")?);
    let warmup = usize_argument(&args, "--warmup", 0)?;
    let measured_runs = usize_argument(&args, "--runs", 1)?;
    if measured_runs == 0 {
        bail!("--runs must be at least one");
    }
    let output_path = optional_argument(&args, "--output").map(PathBuf::from);

    sp1_sdk::setup_logger();
    let started_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let elf_bytes =
        fs::read(&elf_path).with_context(|| format!("read guest ELF {}", elf_path.display()))?;
    let elf = Elf::from(elf_bytes.clone());
    let prepared = prepare_run(relation, &public_path, &witness_path)?;

    let started = Instant::now();
    let client = ProverClient::from_env();
    let prover_backend = env::var("SP1_PROVER").unwrap_or_else(|_| "cpu(default)".to_owned());
    let client_initialization_ms = elapsed_ms(started);

    let started = Instant::now();
    let pk = client.setup(elf.clone())?;
    let setup_ms = elapsed_ms(started);
    let verifying_key_bytes_bincode = bincode::serialize(pk.verifying_key())?.len();

    for index in 0..warmup {
        eprintln!("warmup {}/{} (excluded)", index + 1, warmup);
        run_once(
            &client,
            &elf,
            &pk,
            &prepared.stdin,
            &prepared.expected_public_values,
            0,
        )?;
    }

    let mut runs = Vec::with_capacity(measured_runs);
    for index in 0..measured_runs {
        eprintln!("measured run {}/{}", index + 1, measured_runs);
        runs.push(run_once(
            &client,
            &elf,
            &pk,
            &prepared.stdin,
            &prepared.expected_public_values,
            index + 1,
        )?);
    }

    let output = CampaignOutput {
        schema: "ppmark-five-condition-timing-v1",
        started_unix_ms,
        relation: relation.to_owned(),
        sample_count: prepared.sample_count,
        opening_k: prepared.opening_k,
        sp1_version: "6.3.1",
        proof_mode: "core",
        prover_backend,
        cuda_endpoint_configured: env::var("SP1_CUDA_ENDPOINT").is_ok(),
        elf_path: elf_path.display().to_string(),
        elf_bytes: elf_bytes.len(),
        elf_sha256: hex::encode(Sha256::digest(&elf_bytes)),
        warmup_runs_excluded: warmup,
        measured_runs,
        client_initialization_ms,
        setup_ms,
        verifying_key_bytes_bincode,
        system: system_metadata(),
        execute_ms: distribution(runs.iter().map(|run| run.execute_ms).collect()),
        prove_ms: distribution(runs.iter().map(|run| run.prove_ms).collect()),
        verify_ms: distribution(runs.iter().map(|run| run.verify_ms).collect()),
        serialize_ms: distribution(runs.iter().map(|run| run.serialize_ms).collect()),
        proof_bytes_bincode: distribution(
            runs.iter()
                .map(|run| run.proof_bytes_bincode as f64)
                .collect(),
        ),
        runs,
    };

    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    if let Some(path) = output_path {
        write_output(&path, &output)?;
        eprintln!("campaign JSON -> {}", path.display());
    }
    let drop_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    drop_runtime.block_on(async move {
        drop(pk);
        drop(client);
        tokio::task::yield_now().await;
    });
    Ok(())
}

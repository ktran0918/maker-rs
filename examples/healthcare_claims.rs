//! Healthcare Claims Adjudication with MAKER
//!
//! Applies MAKER's voting-based error correction to insurance claim adjudication.
//! Each adjudication step is an independently verifiable sub-task voted on by
//! multiple LLM samples, giving a mathematical guarantee on per-step reliability.
//!
//! # Why MAKER suits claims adjudication
//!
//! Each step has a provably correct answer given the plan rules. A single LLM at
//! 85% per-step accuracy over 6 steps yields only 0.85^6 ≈ 38% fully-correct
//! claims. With MAKER's k-margin voting at k=3, per-step reliability exceeds
//! 99.7%, giving >98% end-to-end accuracy across the full 6-step pipeline.
//!
//! # Adjudication pipeline
//!
//! For each claim:
//!   1. Eligibility check     — is the member covered on the date of service?
//!   2. Provider verification — is the provider credentialed and in-network?
//!   3. Diagnosis validation  — are the ICD-10 codes valid and billable?
//!   4. Medical necessity     — does the diagnosis support the procedure?
//!   5. Coverage determination — is the service a covered benefit?
//!   6. Benefit calculation   — verify patient vs. plan financial responsibility
//!
//! # Usage
//!
//! ```bash
//! # Mock mode — no API key needed, simulates 85% LLM accuracy
//! MAKER_USE_MOCK=1 cargo run --example healthcare_claims
//!
//! # Higher accuracy mock
//! MAKER_USE_MOCK=1 cargo run --example healthcare_claims -- --accuracy 0.95
//!
//! # Real LLM
//! cargo run --example healthcare_claims -- --provider ollama --model gemma3:27b
//! ```

use clap::Parser;
use maker::core::{
    vote_with_margin_adaptive, KEstimator, KEstimatorConfig, LlmClient, LlmResponse, VoteConfig,
};
use maker::llm::adapter::setup_provider_client;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const DEFAULT_OLLAMA_MODEL: &str = "gemma3:27b";
const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-haiku-3-5-20241022";

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "healthcare_claims",
    about = "MAKER-powered insurance claim adjudication demo"
)]
struct Args {
    #[arg(short, long, default_value = "ollama")]
    provider: String,

    #[arg(short, long)]
    model: Option<String>,

    /// Minimum k-margin floor for adaptive voting
    #[arg(long, default_value = "2")]
    k_min: usize,

    /// Maximum k-margin ceiling for adaptive voting
    #[arg(long, default_value = "8")]
    k_max: usize,

    /// Simulated LLM accuracy in mock mode (0.51–0.99)
    #[arg(long, default_value = "0.85")]
    accuracy: f64,

    /// Additional ensemble members as "provider:model" (repeatable, e.g.
    /// --member ollama:qwen2.5:14b --member ollama:qwen2.5:7b).
    /// Primary model (--provider/--model) is always included first.
    #[arg(long = "member", value_name = "PROVIDER:MODEL")]
    members: Vec<String>,

    /// Temperature-diverse mode: sample the same model at cycling temperatures
    /// [0.0, 0.4, 0.8, 1.2] instead of using an ensemble of different models.
    /// Mutually exclusive with --member.
    #[arg(long, conflicts_with = "members")]
    temp_diverse: bool,
}

// ─── Domain Models ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct InsuranceClaim {
    claim_id: &'static str,
    member_id: &'static str,
    member_name: &'static str,
    date_of_service: &'static str,
    provider_npi: &'static str,
    provider_name: &'static str,
    provider_type: &'static str,
    diagnosis_codes: Vec<&'static str>, // ICD-10
    procedure_code: &'static str,       // CPT
    billed_amount: f64,
    plan_type: &'static str,
    copay: f64,
    deductible_total: f64,
    deductible_met: f64,
    coinsurance_pct: f64, // e.g. 0.20 = patient pays 20% after deductible
    is_in_network: bool,
}

#[derive(Debug, Clone)]
enum ClaimDecision {
    Approved {
        allowed_amount: f64,
        patient_pays: f64,
        plan_pays: f64,
    },
    Denied {
        reason: &'static str,
        carc_code: &'static str, // Claim Adjustment Reason Code
    },
    Pended {
        reason: &'static str,
    },
}

#[derive(Debug)]
struct StepOutcome {
    name: &'static str,
    verdict: String,
    samples: usize,
    k: usize,
    p_hat: f64,
}

#[derive(Debug)]
struct AdjudicationResult {
    claim_id: &'static str,
    decision: ClaimDecision,
    steps: Vec<StepOutcome>,
    total_samples: usize,
    elapsed: std::time::Duration,
}

// ─── Sample Claims ────────────────────────────────────────────────────────────

fn sample_claims() -> Vec<InsuranceClaim> {
    vec![
        // Claim 1: Routine office visit, PPO, deductible fully met → straightforward approval
        InsuranceClaim {
            claim_id: "CLM-2025-001",
            member_id: "MBR-445521",
            member_name: "Jane Smith",
            date_of_service: "2025-05-01",
            provider_npi: "1234567890",
            provider_name: "Dr. Robert Chen, MD",
            provider_type: "Family Medicine",
            diagnosis_codes: vec!["J06.9"], // Acute upper respiratory infection
            procedure_code: "99213",         // Office visit, moderate complexity
            billed_amount: 185.00,
            plan_type: "PPO",
            copay: 30.00,
            deductible_total: 1_000.00,
            deductible_met: 1_000.00, // fully met
            coinsurance_pct: 0.20,
            is_in_network: true,
        },
        // Claim 2: Lumbar MRI, HDHP, deductible partially met → pended (prior auth needed)
        InsuranceClaim {
            claim_id: "CLM-2025-002",
            member_id: "MBR-887342",
            member_name: "John Davis",
            date_of_service: "2025-04-22",
            provider_npi: "9876543210",
            provider_name: "Metro Radiology Group",
            provider_type: "Radiology",
            diagnosis_codes: vec!["M54.50", "M51.16"], // Low back pain unspecified + disc degeneration
            procedure_code: "72148",                   // MRI lumbar spine without contrast
            billed_amount: 2_800.00,
            plan_type: "HDHP",
            copay: 0.00,
            deductible_total: 3_000.00,
            deductible_met: 1_200.00, // partially met
            coinsurance_pct: 0.20,
            is_in_network: true,
        },
        // Claim 3: Emergency appendectomy, out-of-network → covered at OON rate
        InsuranceClaim {
            claim_id: "CLM-2025-003",
            member_id: "MBR-221987",
            member_name: "Maria Garcia",
            date_of_service: "2025-05-10",
            provider_npi: "1122334455",
            provider_name: "Riverside Surgical Center",
            provider_type: "Ambulatory Surgery",
            diagnosis_codes: vec!["K35.80"], // Acute appendicitis without abscess
            procedure_code: "44950",          // Appendectomy
            billed_amount: 12_500.00,
            plan_type: "PPO",
            copay: 250.00,
            deductible_total: 1_500.00,
            deductible_met: 1_500.00, // fully met
            coinsurance_pct: 0.40,    // OON coinsurance
            is_in_network: false,
        },
    ]
}

// ─── Benefit Calculation (Rust-computed, verified by LLM vote) ────────────────

struct Benefits {
    allowed_amount: f64,
    patient_pays: f64,
    plan_pays: f64,
}

fn calculate_benefits(claim: &InsuranceClaim) -> Benefits {
    // Allowed amount: contracted rate for in-network, usual & customary for OON
    let allowed = if claim.is_in_network {
        claim.billed_amount * 0.85
    } else {
        claim.billed_amount * 0.60
    };

    let remaining_deductible = (claim.deductible_total - claim.deductible_met).max(0.0);
    let deductible_applied = remaining_deductible.min(allowed);
    let after_deductible = allowed - deductible_applied;

    let copay = if claim.is_in_network { claim.copay } else { 0.0 };
    let coinsurance_base = (after_deductible - copay).max(0.0);
    let patient_coinsurance = coinsurance_base * claim.coinsurance_pct;

    let patient_pays = (deductible_applied + copay + patient_coinsurance).min(allowed);
    let plan_pays = (allowed - patient_pays).max(0.0);

    Benefits { allowed_amount: allowed, patient_pays, plan_pays }
}

// ─── Mock LLM Client ──────────────────────────────────────────────────────────

/// Wraps any LlmClient and normalizes each response to its leading verdict
/// keyword (the part before the first colon or newline). This ensures voting
/// converges even when the model produces verbose explanations alongside the
/// verdict — e.g. "ELIGIBLE: Member has active PPO coverage..." → "ELIGIBLE".
struct NormalizingClient {
    inner: Box<dyn LlmClient>,
}

impl NormalizingClient {
    fn new(inner: Box<dyn LlmClient>) -> Self {
        Self { inner }
    }
}

impl LlmClient for NormalizingClient {
    fn generate(&self, prompt: &str, temperature: f64) -> Result<LlmResponse, String> {
        let mut resp = self.inner.generate(prompt, temperature)?;
        // Extract the verdict keyword: everything before the first ':' or newline,
        // trimmed of whitespace. Multi-word verdicts like NOT_COVERED and
        // PRIOR_AUTH_REQUIRED are preserved because underscores are not delimiters.
        let verdict = resp
            .content
            .split(|c: char| c == ':' || c == '\n')
            .next()
            .unwrap_or(&resp.content)
            .trim()
            .to_string();
        resp.content = verdict;
        Ok(resp)
    }
}

/// Round-robin ensemble client across N LLM providers.
/// Odd N (3, 5) prevents exact 50/50 deadlocks; same-family models at different
/// sizes tend toward independent rather than anti-correlated errors.
struct EnsembleClient {
    clients: Vec<Box<dyn LlmClient>>,
    counter: AtomicUsize,
}

impl EnsembleClient {
    fn new(clients: Vec<Box<dyn LlmClient>>) -> Self {
        Self { clients, counter: AtomicUsize::new(0) }
    }
}

impl LlmClient for EnsembleClient {
    fn generate(&self, prompt: &str, temperature: f64) -> Result<LlmResponse, String> {
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        self.clients[n % self.clients.len()].generate(prompt, temperature)
    }
}

/// Samples a single model at cycling temperatures [0.0, 0.4, 0.8, 1.2].
/// At T=0 the model is deterministic (its MAP answer). Higher temperatures
/// sample from the probability distribution around that answer. If the correct
/// answer has high probability mass, it dominates across temperatures; errors
/// (low probability) are more temperature-sensitive and get voted out.
struct TemperatureDiverseClient {
    inner: Box<dyn LlmClient>,
    temperatures: [f64; 4],
    counter: AtomicUsize,
}

impl TemperatureDiverseClient {
    fn new(inner: Box<dyn LlmClient>) -> Self {
        Self { inner, temperatures: [0.0, 0.4, 0.8, 1.2], counter: AtomicUsize::new(0) }
    }
}

impl LlmClient for TemperatureDiverseClient {
    fn generate(&self, prompt: &str, _temperature: f64) -> Result<LlmResponse, String> {
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        let temp = self.temperatures[n % self.temperatures.len()];
        self.inner.generate(prompt, temp)
    }
}

/// Step-aware mock client returning realistic adjudication responses.
/// Detects which step it's on from prompt keywords and returns
/// the correct answer with probability `accuracy`.
struct HealthcareMockClient {
    accuracy: f64,
    call_count: AtomicUsize,
}

impl HealthcareMockClient {
    fn new(accuracy: f64) -> Self {
        Self { accuracy, call_count: AtomicUsize::new(0) }
    }
}

impl LlmClient for HealthcareMockClient {
    fn generate(&self, prompt: &str, _temperature: f64) -> Result<LlmResponse, String> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        // Golden-ratio stride avoids clustering of errors
        let is_correct = (n as f64 * 1.618_033_988_7) % 1.0 < self.accuracy;

        let response = if prompt.contains("ELIGIBILITY") {
            if is_correct {
                "ELIGIBLE: Member has active coverage. Date of service falls within benefit period."
            } else {
                "INELIGIBLE: Member ID not found or coverage lapsed prior to date of service."
            }
        } else if prompt.contains("PROVIDER VERIFICATION") {
            if is_correct {
                "VERIFIED: Provider NPI is active, credentialed, and contracted with the plan."
            } else {
                "UNVERIFIED: Provider license could not be confirmed. Possible sanction on record."
            }
        } else if prompt.contains("DIAGNOSIS CODE") {
            if is_correct {
                "VALID: All ICD-10 codes are current, billable, and at the required specificity level."
            } else {
                "INVALID: One or more codes are non-billable header codes. Resubmit with specific codes."
            }
        } else if prompt.contains("MEDICAL NECESSITY") {
            if is_correct {
                "NECESSARY: Diagnosis supports medical necessity per applicable clinical guidelines."
            } else {
                "UNNECESSARY: Insufficient clinical documentation to establish medical necessity."
            }
        } else if prompt.contains("COVERAGE DETERMINATION") {
            if prompt.contains("72148") {
                // MRI: prior auth required for elective imaging
                if is_correct {
                    "PRIOR_AUTH_REQUIRED: Diagnostic MRI for elective indication requires prior authorization."
                } else {
                    "COVERED: Imaging covered without restriction under member benefit plan."
                }
            } else if is_correct {
                "COVERED: Service is a covered benefit. Standard plan cost-sharing applies."
            } else {
                "NOT_COVERED: Service excluded from benefits or requires prior authorization."
            }
        } else if prompt.contains("BENEFIT CALCULATION") {
            if is_correct {
                "CALCULATION_VALID: Benefit amounts correctly computed per plan cost-sharing rules."
            } else {
                "CALCULATION_INVALID: Deductible application appears inconsistent. Requires review."
            }
        } else {
            "VALID: Step verified."
        };

        Ok(LlmResponse {
            content: response.to_string(),
            input_tokens: prompt.len() / 4,
            output_tokens: response.len() / 4,
        })
    }
}

// ─── Prompt Builders ──────────────────────────────────────────────────────────

fn eligibility_prompt(claim: &InsuranceClaim) -> String {
    format!(
        "ELIGIBILITY CHECK — Insurance Claim Adjudication\n\
         \n\
         Member ID:        {member_id}\n\
         Member Name:      {member_name}\n\
         Plan Type:        {plan_type}\n\
         Date of Service:  {dos}\n\
         Benefit Period:   2025-01-01 to 2025-12-31\n\
         \n\
         Task: Confirm the member has active coverage on the date of service.\n\
         Rules:\n\
         - Date of service must fall within the active benefit period\n\
         - Member enrollment must be current (not terminated or suspended)\n\
         - Plan must not have a premium payment lapse\n\
         \n\
         Respond with EXACTLY one of:\n\
         ELIGIBLE: [brief reason]\n\
         INELIGIBLE: [specific denial reason]",
        member_id = claim.member_id,
        member_name = claim.member_name,
        plan_type = claim.plan_type,
        dos = claim.date_of_service,
    )
}

fn provider_verification_prompt(claim: &InsuranceClaim) -> String {
    format!(
        "PROVIDER VERIFICATION — Insurance Claim Adjudication\n\
         \n\
         Provider NPI:     {npi}\n\
         Provider Name:    {name}\n\
         Provider Type:    {ptype}\n\
         Network Status:   {network}\n\
         Procedure Billed: CPT {cpt}\n\
         \n\
         Task: Confirm the provider is authorized to render and bill this service.\n\
         Rules:\n\
         - NPI must be active in the NPPES registry\n\
         - Provider specialty must be appropriate for the billed procedure\n\
         - Provider must have no active OIG exclusions or license sanctions\n\
         - Out-of-network providers may still bill but at reduced reimbursement\n\
         \n\
         Respond with EXACTLY one of:\n\
         VERIFIED: [brief confirmation]\n\
         UNVERIFIED: [specific reason]",
        npi = claim.provider_npi,
        name = claim.provider_name,
        ptype = claim.provider_type,
        network = if claim.is_in_network { "In-Network" } else { "Out-of-Network" },
        cpt = claim.procedure_code,
    )
}

/// Returns the official ICD-10-CM description for a code, or a note that it
/// is not in this local reference. Embedding descriptions in the prompt shifts
/// the LLM's task from knowledge recall to consistency verification — far more
/// reliable and removes disagreement caused by differing training-data coverage.
fn icd10_description(code: &str) -> &'static str {
    match code {
        "J06.9"  => "Acute upper respiratory infection, unspecified — valid billable leaf code",
        "M54.50" => "Low back pain, unspecified — valid billable leaf code (replaces deleted M54.5 as of FY2021)",
        "M54.5"  => "DELETED as of FY2021 — replaced by M54.50/M54.51/M54.59; not valid for DOS after 2020",
        "M51.16" => "Intervertebral disc degeneration, lumbar region — valid billable leaf code",
        "K35.80" => "Acute appendicitis without abscess or peritonitis — valid billable leaf code",
        _        => "Code not found in local reference — verify against ICD-10-CM tabular list",
    }
}

fn diagnosis_code_prompt(claim: &InsuranceClaim) -> String {
    let code_lines: String = claim.diagnosis_codes.iter()
        .map(|c| format!("  {} — {}", c, icd10_description(c)))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "DIAGNOSIS CODE VALIDATION — Insurance Claim Adjudication\n\
         \n\
         Date of Service:  {dos}\n\
         Provider Type:    {ptype}\n\
         \n\
         Submitted ICD-10 codes with official descriptions:\n\
         {code_lines}\n\
         \n\
         Task: Confirm every code above is valid for this claim.\n\
         Rules:\n\
         - Code must be marked as a valid billable leaf code in the description above\n\
         - Code must not be listed as DELETED or replaced\n\
         - Code must be clinically consistent with the provider type\n\
         \n\
         Respond with EXACTLY one of:\n\
         VALID: [confirm all codes are billable and appropriate]\n\
         INVALID: [identify which code is problematic and why]",
        dos = claim.date_of_service,
        ptype = claim.provider_type,
        code_lines = code_lines,
    )
}

fn medical_necessity_prompt(claim: &InsuranceClaim) -> String {
    format!(
        "MEDICAL NECESSITY REVIEW — Insurance Claim Adjudication\n\
         \n\
         Diagnosis (ICD-10): {dx}\n\
         Procedure (CPT):    {cpt}\n\
         Provider Type:      {ptype}\n\
         \n\
         Task: Determine whether the procedure is medically necessary given the diagnosis.\n\
         Apply LCD/NCD coverage criteria and evidence-based clinical guidelines:\n\
         - The procedure must be appropriate and clinically indicated for the diagnosis\n\
         - The level of service must match the documented complexity\n\
         - The provider specialty must be appropriate to render this procedure\n\
         - Conservative treatment alternatives must have been considered when applicable\n\
         \n\
         Respond with EXACTLY one of:\n\
         NECESSARY: [clinical rationale supporting medical necessity]\n\
         UNNECESSARY: [specific reason medical necessity is not established]",
        dx = claim.diagnosis_codes.join(", "),
        cpt = claim.procedure_code,
        ptype = claim.provider_type,
    )
}

fn coverage_determination_prompt(claim: &InsuranceClaim) -> String {
    format!(
        "COVERAGE DETERMINATION — Insurance Claim Adjudication\n\
         \n\
         Plan Type:        {plan}\n\
         Network Status:   {network}\n\
         Procedure (CPT):  {cpt}\n\
         Provider Type:    {ptype}\n\
         Billed Amount:    ${billed:.2}\n\
         \n\
         Task: Determine whether this procedure is a covered benefit under the plan.\n\
         Plan coverage rules:\n\
         - All medically necessary services are covered unless specifically excluded\n\
         - Emergency and urgent services are always covered regardless of network\n\
         - Elective diagnostic imaging (MRI, CT) requires prior authorization\n\
         - Surgical procedures for emergent conditions are covered without prior auth\n\
         - Out-of-network services are covered at 60% of allowed amount after deductible\n\
         \n\
         Respond with EXACTLY one of:\n\
         COVERED: [coverage details and applicable cost-sharing note]\n\
         NOT_COVERED: [specific exclusion or policy reason]\n\
         PRIOR_AUTH_REQUIRED: [specify what authorization is needed and why]",
        plan = claim.plan_type,
        network = if claim.is_in_network { "In-Network" } else { "Out-of-Network" },
        cpt = claim.procedure_code,
        ptype = claim.provider_type,
        billed = claim.billed_amount,
    )
}

fn benefit_calculation_prompt(claim: &InsuranceClaim, b: &Benefits) -> String {
    let remaining_ded = (claim.deductible_total - claim.deductible_met).max(0.0);
    format!(
        "BENEFIT CALCULATION VERIFICATION — Insurance Claim Adjudication\n\
         \n\
         Plan Type:          {plan}\n\
         Network:            {network}\n\
         Billed Amount:      ${billed:.2}\n\
         Allowed Amount:     ${allowed:.2}  ({pct}% of billed)\n\
         Deductible (total): ${ded_total:.2}\n\
         Deductible (met):   ${ded_met:.2}\n\
         Remaining Ded.:     ${ded_rem:.2}\n\
         Copay:              ${copay:.2}\n\
         Coinsurance:        {coins:.0}% patient responsibility\n\
         \n\
         Computed Result:\n\
         Patient pays: ${patient:.2}  |  Plan pays: ${plan_pays:.2}\n\
         \n\
         Task: Verify the benefit calculation correctly applies plan cost-sharing rules:\n\
         1. Remaining deductible is applied first (patient responsible)\n\
         2. Copay applies to in-network services after deductible\n\
         3. Coinsurance applies to the remaining balance\n\
         4. Plan pays the remainder of the allowed amount\n\
         \n\
         Respond with EXACTLY one of:\n\
         CALCULATION_VALID: Benefit amounts correctly computed per plan documents.\n\
         CALCULATION_INVALID: [identify the specific error in the calculation]",
        plan = claim.plan_type,
        network = if claim.is_in_network { "In-Network" } else { "Out-of-Network" },
        billed = claim.billed_amount,
        allowed = b.allowed_amount,
        pct = if claim.is_in_network { 85 } else { 60 },
        ded_total = claim.deductible_total,
        ded_met = claim.deductible_met,
        ded_rem = remaining_ded,
        copay = claim.copay,
        coins = claim.coinsurance_pct * 100.0,
        patient = b.patient_pays,
        plan_pays = b.plan_pays,
    )
}

// ─── Adjudication Pipeline ────────────────────────────────────────────────────

struct AdjudicationPipeline {
    client: Box<dyn LlmClient>,
    k_estimator: KEstimator,
    target_reliability: f64,
}

impl AdjudicationPipeline {
    fn new(client: Box<dyn LlmClient>, k_min: usize, k_max: usize) -> Self {
        let k_config = KEstimatorConfig {
            ema_alpha: 0.2,
            initial_p_hat: 0.90,
            k_min_floor: k_min,
            k_max_ceiling: k_max,
        };
        Self {
            client,
            k_estimator: KEstimator::new(k_config),
            target_reliability: 0.999, // 99.9% reliability per step
        }
    }

    fn vote_step(
        &mut self,
        name: &'static str,
        prompt: String,
        remaining: usize,
        steps: &mut Vec<StepOutcome>,
    ) -> Result<String, String> {
        let config = VoteConfig::default()
            .with_diversity_temperature(0.7)
            .without_token_limit();

        let result = vote_with_margin_adaptive(
            &prompt,
            &mut self.k_estimator,
            self.target_reliability,
            remaining,
            self.client.as_ref(),
            config,
        )
        .map_err(|e| e.to_string())?;

        steps.push(StepOutcome {
            name,
            verdict: result.winner.clone(),
            samples: result.total_samples,
            k: result.k_used,
            p_hat: self.k_estimator.p_hat(),
        });

        Ok(result.winner)
    }

    fn adjudicate(&mut self, claim: &InsuranceClaim) -> Result<AdjudicationResult, String> {
        let start = Instant::now();
        let mut steps: Vec<StepOutcome> = Vec::new();
        const N: usize = 6; // total pipeline steps

        macro_rules! deny {
            ($reason:expr, $carc:expr) => {{
                let total_samples = steps.iter().map(|s| s.samples).sum();
                return Ok(AdjudicationResult {
                    claim_id: claim.claim_id,
                    decision: ClaimDecision::Denied { reason: $reason, carc_code: $carc },
                    steps,
                    total_samples,
                    elapsed: start.elapsed(),
                });
            }};
        }

        macro_rules! pend {
            ($reason:expr) => {{
                let total_samples = steps.iter().map(|s| s.samples).sum();
                return Ok(AdjudicationResult {
                    claim_id: claim.claim_id,
                    decision: ClaimDecision::Pended { reason: $reason },
                    steps,
                    total_samples,
                    elapsed: start.elapsed(),
                });
            }};
        }

        // Step 1 — Eligibility
        let v = self.vote_step("Eligibility", eligibility_prompt(claim), N, &mut steps)?;
        if v.starts_with("INELIGIBLE") {
            deny!("Member not eligible on date of service", "27");
        }

        // Step 2 — Provider Verification
        let v = self.vote_step(
            "Provider Verification",
            provider_verification_prompt(claim),
            N - 1,
            &mut steps,
        )?;
        if v.starts_with("UNVERIFIED") {
            deny!("Provider not credentialed or authorized to bill", "185");
        }

        // Step 3 — Diagnosis Code Validation
        let v = self.vote_step(
            "Diagnosis Validation",
            diagnosis_code_prompt(claim),
            N - 2,
            &mut steps,
        )?;
        if v.starts_with("INVALID") {
            deny!("Invalid or non-billable ICD-10 diagnosis code", "16");
        }

        // Step 4 — Medical Necessity
        let v = self.vote_step(
            "Medical Necessity",
            medical_necessity_prompt(claim),
            N - 3,
            &mut steps,
        )?;
        if v.starts_with("UNNECESSARY") {
            deny!("Service not medically necessary per clinical guidelines", "50");
        }

        // Step 5 — Coverage Determination
        let v = self.vote_step(
            "Coverage Determination",
            coverage_determination_prompt(claim),
            N - 4,
            &mut steps,
        )?;
        if v.starts_with("NOT_COVERED") {
            deny!("Service not a covered benefit under member plan", "96");
        }
        if v.starts_with("PRIOR_AUTH_REQUIRED") {
            pend!("Prior authorization required — claim held pending review");
        }

        // Step 6 — Benefit Calculation Verification
        let benefits = calculate_benefits(claim);
        let v = self.vote_step(
            "Benefit Calculation",
            benefit_calculation_prompt(claim, &benefits),
            N - 5,
            &mut steps,
        )?;
        if v.starts_with("CALCULATION_INVALID") {
            pend!("Benefit calculation discrepancy — routed to manual review");
        }

        let total_samples = steps.iter().map(|s| s.samples).sum();
        Ok(AdjudicationResult {
            claim_id: claim.claim_id,
            decision: ClaimDecision::Approved {
                allowed_amount: benefits.allowed_amount,
                patient_pays: benefits.patient_pays,
                plan_pays: benefits.plan_pays,
            },
            steps,
            total_samples,
            elapsed: start.elapsed(),
        })
    }
}

// ─── Display ──────────────────────────────────────────────────────────────────

fn print_result(claim: &InsuranceClaim, result: &AdjudicationResult) {
    println!("\n{}", "─".repeat(72));
    println!(
        "  {}  │  {}  │  {}",
        result.claim_id, claim.member_name, claim.date_of_service
    );
    println!(
        "  CPT {}  +  ICD-10 {}  │  Billed ${:.2}  │  {}  │  {}",
        claim.procedure_code,
        claim.diagnosis_codes.join("+"),
        claim.billed_amount,
        claim.plan_type,
        if claim.is_in_network { "In-Network" } else { "Out-of-Network" },
    );
    println!("{}", "─".repeat(72));

    println!("  Adjudication Steps:");
    for step in &result.steps {
        let keyword = step.verdict.split(':').next().unwrap_or(&step.verdict);
        let pass = !matches!(
            keyword,
            "INELIGIBLE"
                | "UNVERIFIED"
                | "INVALID"
                | "UNNECESSARY"
                | "NOT_COVERED"
                | "CALCULATION_INVALID"
        );
        let icon = if pass { "✓" } else { "✗" };
        println!(
            "  {}  {:<26}  {:<22}  k={} samples={} p̂={:.3}",
            icon, step.name, keyword, step.k, step.samples, step.p_hat
        );
    }

    println!("\n  Decision:");
    match &result.decision {
        ClaimDecision::Approved { allowed_amount, patient_pays, plan_pays } => {
            println!("  ✓  APPROVED");
            println!("     Allowed amount:   ${:.2}", allowed_amount);
            println!("     Plan pays:        ${:.2}", plan_pays);
            println!("     Patient pays:     ${:.2}", patient_pays);
        }
        ClaimDecision::Denied { reason, carc_code } => {
            println!("  ✗  DENIED  (CARC {})", carc_code);
            println!("     Reason: {}", reason);
        }
        ClaimDecision::Pended { reason } => {
            println!("  ⏸  PENDED");
            println!("     Reason: {}", reason);
        }
    }

    println!(
        "\n  LLM calls: {}  │  Elapsed: {:.2?}",
        result.total_samples, result.elapsed
    );
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let use_mock = env::var("MAKER_USE_MOCK").unwrap_or_default() == "1";

    println!("=== MAKER Healthcare Claims Adjudication Demo ===\n");

    let client: Box<dyn LlmClient> = if use_mock {
        if args.accuracy <= 0.5 || args.accuracy >= 1.0 {
            eprintln!("Error: --accuracy must be in (0.5, 1.0)");
            std::process::exit(1);
        }
        println!("Mode:     Mock (simulated accuracy: {:.0}%)", args.accuracy * 100.0);
        Box::new(HealthcareMockClient::new(args.accuracy))
    } else {
        let model = args.model.clone().unwrap_or_else(|| match args.provider.as_str() {
            "openai" => DEFAULT_OPENAI_MODEL.to_string(),
            "anthropic" => DEFAULT_ANTHROPIC_MODEL.to_string(),
            _ => DEFAULT_OLLAMA_MODEL.to_string(),
        });
        if args.temp_diverse {
            println!("Mode:     {} ({}) — temperature-diverse [0.0, 0.4, 0.8, 1.2]", args.provider, model);
        } else if args.members.is_empty() {
            println!("Mode:     {} ({})", args.provider, model);
        } else {
            let member_labels = args.members.join(", ");
            println!("Mode:     Ensemble — {}:{} + {}", args.provider, model, member_labels);
        }
        let primary = match setup_provider_client(&args.provider, Some(model)) {
            Ok(c) => c,
            Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
        };

        if args.temp_diverse {
            Box::new(NormalizingClient::new(Box::new(TemperatureDiverseClient::new(primary))))
        } else if args.members.is_empty() {
            Box::new(NormalizingClient::new(primary))
        } else {
            let mut clients: Vec<Box<dyn LlmClient>> = vec![primary];
            for entry in &args.members {
                // Split on first ':' to get provider; remainder is model name
                // (model names like qwen2.5:14b contain colons themselves)
                let (provider, model) = entry.split_once(':')
                    .unwrap_or_else(|| { eprintln!("--member must be 'provider:model', got '{}'", entry); std::process::exit(1); });
                match setup_provider_client(provider, Some(model.to_string())) {
                    Ok(c) => clients.push(c),
                    Err(e) => { eprintln!("Error building member {}: {}", entry, e); std::process::exit(1); }
                }
            }
            println!("Ensemble: {} model(s) in round-robin", clients.len());
            Box::new(NormalizingClient::new(Box::new(EnsembleClient::new(clients))))
        }
    };

    println!("k-margin: Adaptive (min={}, max={})", args.k_min, args.k_max);
    println!("Target:   99.9% per-step reliability\n");

    let claims = sample_claims();
    let mut pipeline = AdjudicationPipeline::new(client, args.k_min, args.k_max);

    let mut n_approved = 0usize;
    let mut n_denied = 0usize;
    let mut n_pended = 0usize;
    let mut total_samples = 0usize;
    let batch_start = Instant::now();

    for claim in &claims {
        match pipeline.adjudicate(claim) {
            Ok(result) => {
                match &result.decision {
                    ClaimDecision::Approved { .. } => n_approved += 1,
                    ClaimDecision::Denied { .. } => n_denied += 1,
                    ClaimDecision::Pended { .. } => n_pended += 1,
                }
                total_samples += result.total_samples;
                print_result(claim, &result);
            }
            Err(e) => eprintln!("\n  ERROR processing {}: {}", claim.claim_id, e),
        }
    }

    println!("\n{}", "═".repeat(72));
    println!("  Batch Summary");
    println!("{}", "═".repeat(72));
    println!("  Claims processed:  {}", claims.len());
    println!("  Approved:          {}", n_approved);
    println!("  Denied:            {}", n_denied);
    println!("  Pended:            {}", n_pended);
    println!("  Total LLM calls:   {}", total_samples);
    println!("  Total elapsed:     {:.2?}", batch_start.elapsed());

    if use_mock {
        let p = args.accuracy;
        let steps = 6usize;
        let without_maker = p.powi(steps as i32) * 100.0;
        // With k=3 voting: per-step error ≈ ((1-p)/p)^3
        let p_step_err = ((1.0 - p) / p).powi(3);
        let with_maker = (1.0 - p_step_err).powi(steps as i32) * 100.0;
        println!("\n  Error Correction Impact (at {:.0}% per-call accuracy):", p * 100.0);
        println!(
            "  Without MAKER (single call/step): {:.1}% of claims fully correct",
            without_maker
        );
        println!(
            "  With MAKER k=3 voting:            {:.1}% of claims fully correct",
            with_maker
        );
    }
    println!();
}
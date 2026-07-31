Title: Continuum Memory Architecture (CMA) for Clinical Search: A Protocol for Evaluating Intent-Aware Retrieval to Improve Longitudinal Chart Review and Point-of-Care Decision Support

Authors: [Redacted — draft]
Affiliations: [Redacted — draft]

Corresponding author: [Name, email, institution]

Word count: ~4500 (draft)

1. Abstract

Background: Clinicians reviewing longitudinal electronic health records (EHRs) navigate non-linear information trajectories. Standard retrieval systems suffer latent-context "pollution" when prior search context degrades relevance after abrupt topic shifts. The Continuum Memory Architecture (CMA) is a recently described intent-tracking and forecasting framework that uses curvature-aware gating, joint-embedding predictive models (JEPA), and symmetric positive definite (SPD) intent representations to suppress stale context and proactively fetch likely next-query results.

Objective: To present a clinical study protocol to evaluate CMA-augmented EHR retrieval versus standard session-based search for task performance, cognitive load, and decision-support timeliness in simulation-based chart review settings using publicly available clinical data.

Methods: We will perform a randomized, crossover, simulation-based benchmark evaluation using publicly available clinical datasets and synthetic EHR cases. The benchmark includes 60 complex chart-review cases with abrupt topic shifts (cardio → renal → medication reconciliation). Primary outcomes are time-to-correct-information (seconds) and retrieval accuracy (proportion correct). Secondary outcomes include system latency and expert annotation review. Sample size is powered to detect a 20% reduction in median time-to-answer (alpha=0.05, power=0.8). Analysis uses mixed-effects models accounting for task clustering and case complexity. Ethical considerations focus on data provenance and privacy for publicly available clinical datasets.

Conclusions: This protocol evaluates CMA's promise for reducing latent-context interference and enabling proactive retrieval in clinical settings. If successful, CMA could shorten critical decision times and reduce cognitive burden, but deployment requires strict guardrails to mitigate prediction-induced bias.

Keywords: Continuum Memory Architecture, EHR search, clinical decision support, intent forecasting, JEPA, Curvature-Aware Interference

2. Introduction

2.1 Background and current knowledge

Clinicians routinely perform longitudinal chart review when evaluating inpatients and outpatients. Searching across medications, laboratory trends, imaging reports, and consult notes is multi-turn and often non-linear: clinicians pivot between organ systems and problem domains depending on evolving hypotheses and new findings. Conventional retrieval systems maintain session context naively (e.g., last-query expansion or fixed history windows), which can cause latent-context pollution — previously relevant context continues to influence retrieval despite an abrupt shift in intent. The Continuum Memory Architecture (CMA) was introduced to model evolving intent trajectories and predict near-future intent using a combination of SPD geometry for intent representation, a Curvature-Aware Interference (CAI) gate to suppress stale context at sharp curvature changes, and Joint Embedding Predictive Architecture (JEPA) modules to forecast upcoming intent states. These components together promise low-latency prefetch and context-aware suppression that could improve relevance and responsiveness in multi-turn search tasks.

2.2 Research gap

While CMA has been evaluated on web-search benchmarks and general multi-turn retrieval tasks, its utility in safety-critical clinical environments has not been demonstrated. Clinical workflows introduce high stakes (patient safety), domain-specific ontologies (SNOMED, MeSH, LOINC), and regulatory and ethical constraints. A rigorous clinical evaluation is needed to quantify CMA's effects on task performance, cognitive load, accuracy of retrieved facts, and unintended effects such as confirmation bias introduced by proactive suggestions.

2.3 Rationale

Shortening clinician time-to-information and reducing cognitive switching cost can plausibly improve patient care timeliness and clinician efficiency. CMA's CAI gate can reduce interference after abrupt topic pivots (e.g., cardiology→nephrology), while JEPA-based prefetching can materially reduce retrieval latency in emergent situations where seconds matter. Evaluating CMA in a controlled clinical simulation using publicly available data will provide evidence for benefits and potential harms, and inform necessary guardrails.

2.4 Objective and hypotheses

Primary objective: Compare CMA-augmented EHR search to standard EHR search for clinician time-to-correct-information and retrieval accuracy during longitudinal chart review tasks.

Hypotheses:
- H1: CMA reduces median time-to-correct-information by ≥20% versus standard search.
- H2: CMA increases retrieval accuracy (proportion correct) or maintains non-inferiority within a 5% margin.
- H3: CMA reduces subjective cognitive load (NASA-TLX) and improves perceived usefulness, without increasing erroneous actions attributable to proactive suggestions.

3. Methods

3.1 Study design

Randomized, within-subject crossover benchmark design with simulation-based data. Each condition includes 8 clinical vignettes, with vignette order randomized and balanced. A washout task (5-minute unrelated quiz) separates conditions.

3.2 Study setting

Academic tertiary-care hospital EHR testbed environment using de-identified synthetic patient charts constructed from real EHR distributions and publicly available MIMIC-derived synthetic cases. Simulation lab with workstation recording of screen, keystrokes, and system logs for latency and retrieval traces.

3.3 Data sources and benchmark cases

The evaluation uses publicly available, de-identified clinical datasets and synthetic chart cases. Case selection criteria include:
- Complex longitudinal clinical cases with multi-domain content (notes, labs, medications, imaging reports)
- Tasks requiring at least three topic pivots to reflect realistic information-seeking behavior
- Publicly available or synthetic data sources that are standard in clinical retrieval research

No new prospective clinician recruitment is required.

3.4 Benchmark case selection

Assumptions: mean baseline time-to-correct-information = 120s (SD 60s) per task; target reduction 20% (24s absolute); within-task correlation rho=0.5. For paired t-test approximation, two-sided alpha=0.05, power=0.8, required n≈52 benchmark tasks. We selected 60 cases to ensure sufficient power and to account for any unusable cases or data anomalies.

3.5 Randomization and blinding

Task order was randomized using computer-generated block randomization (blocks of 4) to control for order effects. Outcome scoring was adjudicated by blinded expert reviewers using de-identified log recordings.

3.6 Intervention (CMA-enabled system)

CMA components integrated into the EHR test environment:
- Session intent encoder producing SPD representations of active intent.
- CAI gate that monitors curvature of intent trajectories and attenuates prior-context contributions when curvature exceeds a learned threshold.
- JEPA module producing short-horizon predictions of next intent; predicted top-K retrievals are prefetched and surfaced in a 'suggested items' panel with provenance and confidence scores.

Control: Standard EHR search (baseline session-based retrieval with recency-weighted context window, no prediction; identical UI except absence of proactive suggestions panel).

3.7 Data collection procedures

For each task/vignette: start time, search queries, time of first correct retrieval (as defined by adjudicator), final answer submission, keystroke and click logs, system latency; questionnaire responses post-session (NASA-TLX, SUS, perceived usefulness). All logs stored on secure hospital research servers in de-identified form.

3.8 Outcome measures

Primary outcomes:
- Time-to-correct-information (seconds) per task.
- Retrieval accuracy: proportion of tasks with correct information retrieved within allowed time (300s).

Secondary outcomes:
- Cognitive load: NASA-TLX composite score.
- Perceived usability and trust: SUS and brief survey.
- System latency: median retrieval latency (ms).
- Error analysis: rate and nature of incorrect actions plausibly influenced by proactive suggestions.

3.9 Statistical analysis

Primary analyses use linear mixed-effects models for time-to-event (log-transformed if skewed) with fixed effects for condition (CMA vs control), period, and vignette type, and random intercepts for task. Retrieval accuracy analyzed with mixed-effects logistic regression. Effect sizes reported with 95% confidence intervals and two-sided p-values. Pre-specified subgroup analyses: specialty, years of experience, vignette complexity. Missing data: multiple imputation for questionnaire outcomes; task-level missingness handled via sensitivity analyses (complete-case and best/worst-case bounds).

3.10 Data provenance and privacy

This evaluation uses publicly available de-identified datasets and synthetic cases, ensuring no new patient-protected health information is introduced. All data sources are managed under standard privacy and data-use agreements. System logs and benchmark results are stored securely with role-based access controls.

3.11 Reporting guidelines

This evaluation follows reporting guidance for computational clinical retrieval evaluations and clinical informatics simulation studies. EQUATOR Network resources and relevant journal guidelines will guide final reporting.

4. Results (Planned analyses and reporting — protocol paper)

Because this is a protocol and pilot-ready design, no primary trial results are yet available. The following describes planned presentation:

4.1 Benchmark case flow

Flow diagram showing benchmark cases selected, randomized to conditions, analyzed, and any cases excluded due to data issues.

4.2 Baseline characteristics

Table 1: Benchmark case characteristics and data provenance (source dataset, complexity, domain coverage).

4.3 Primary and secondary outcomes

Table 2: Primary outcomes (median time-to-correct-information, interquartile range) by condition; mixed-effects model estimates with effect sizes, 95% CI, and p-values.

Table 3: Retrieval accuracy, NASA-TLX, SUS, median latency.

Figures:
- Figure 1: Benchmark flow diagram.
- Figure 2: Violin/box plots of time-to-correct-information by condition and vignette type.
- Figure 3: Example intent trajectory visualizations (SPD manifold) showing CAI gating events and JEPA prediction timelines for representative case.

4.4 Safety and error analyses

Adjudicated descriptions of any instances where proactive suggestions plausibly contributed to erroneous actions; rates will be compared between conditions.

5. Discussion

5.1 Main findings (anticipated)

We anticipate CMA will reduce time-to-information and perceived cognitive load while maintaining or improving retrieval accuracy. JEPA-backed prefetching is expected to reduce median retrieval latency significantly in high-switching vignettes.

5.2 Comparison with previous studies

This study extends prior CMA evaluations on web-search benchmarks into clinical workflows, aligning with literature showing that retrieval latency and context management impact clinician efficiency and error rates.

5.3 Mechanisms and explanations

Curvature-aware gating suppresses stale context after sharp intent shifts, reducing spurious term re-weighting. JEPA reduces the cold-start retrieval penalty by warming caches and surfacing relevant content with provenance and confidence scores to support clinician judgment.

5.4 Strengths

- Randomized crossover benchmark design controlling for within-case variability.
- Use of synthetic and MIMIC-derived cases to balance realism and privacy.
- Detailed logging for reproducible analysis.

5.5 Limitations

- Simulation environment may not fully capture real-world EHR complexity and interruptions.
- Partial inability to blind reviewers to UI differences may introduce assessment bias.
- Ethical risks of proactive suggestions (confirmation bias) require careful monitoring.

5.6 Clinical implications

If validated, CMA could be integrated into EHR search modules to improve clinician efficiency during chart review and support rapid decision-making in acute care. Deployment must include transparency, provenance, and simple opt-out mechanisms for predictive suggestions.

5.7 Future research

Larger multi-center trials, longitudinal assessments of diagnostic outcomes, and human factors studies on trust calibration and interface design.

6. Conclusion

This protocol outlines a rigorous evaluation of the Continuum Memory Architecture in clinical search tasks. CMA's curvature-aware suppression and predictive prefetching may reduce time-to-information and clinician cognitive burden, but careful ethics oversight and deployment guardrails are essential.

7. References (selected — verify formatting and DOIs)

1. [CMA paper]. MECO_2026_Corrected_paper_14.pdf (internal manuscript in workspace).
2. Moher D, Liberati A, Tetzlaff J, Altman DG; PRISMA Group. Preferred reporting items for systematic reviews and meta-analyses: the PRISMA statement. PLoS Med. 2009;6(7):e1000097. doi:10.1371/journal.pmed.1000097
3. Equator Network. Reporting guidance for computational clinical retrieval evaluations and simulation studies. [Online resource].
4. von Elm E, Altman DG, Egger M, Pocock SJ, Gøtzsche PC, Vandenbroucke JP; STROBE Initiative. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. Lancet. 2007;370(9596):1453-1457. doi:10.1016/S0140-6736(07)61602-X
5. Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. Nat Med. 2019;25:44–56. doi:10.1038/s41591-018-0300-7
6. Amann J, Blasimme A, Vayena E, Frey D, Madai VI. Explainability for artificial intelligence in healthcare: a multidisciplinary perspective. BMC Med Inform Decis Mak. 2020;20(1):310. doi:10.1186/s12911-020-01214-5

8. Tables and Figures (placeholders)

Table 1. Baseline characteristics (template)

| Variable | CMA first (n=30) | Control first (n=30) | Total (n=60) |
|---|---:|---:|---:|
| Age, mean (SD) |  |  |  |
| Sex, n (%) |  |  |  |
| Specialty, n (%) |  |  |  |
| Years since graduation, median (IQR) |  |  |  |

Table 2. Primary outcomes (template)

| Outcome | CMA | Control | Difference (95% CI) | p-value |
|---|---:|---:|---:|---:|
| Time-to-correct-information, median (IQR) |  |  |  |  |
| Retrieval accuracy, % |  |  |  |  |

Figure 1. Benchmark case flow diagram (placeholder)

Figure 2. Time-to-correct-information by condition (violin/boxplot)

Figure 3. Representative intent trajectory visualization (SPD manifold) with CAI gating highlights and JEPA prediction window

9. Supplementary material (optional)

- Supplementary Appendix A: Full vignette text and scoring rubric
- Supplementary Appendix B: CMA model hyperparameters and training data provenance
- Supplementary Appendix C: Statistical analysis code (R/Python) and simulated sample analysis outputs

Acknowledgements: [TBD]

Funding: [TBD]

Conflicts of interest: [TBD]

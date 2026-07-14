# Step 4 Coherence Spot-Check: Candidate Reversal Features

Completes the outstanding half of CLAUDE.md Step 4 (Methodology section) for the 4 features
flagged by `signed_reanalysis.md` as non-trivial suppression "reversals":
`data/features/spot_check_reversal_candidates.json` = `2_13823`, `7_12166`, `7_6944`, `14_6669`.

**Method:** manual read of the cached generations in
`data/generated_texts/gemma_2b/gemma2_2b_generations_cache_{10,neg2,neg10,neg20}_0.json`
(6 of the cached prompts per feature/magnitude shown below; all generations for a feature/magnitude
pair were read, not just the excerpts quoted). No LLM judge or perplexity metric was used — this is
the qualitative "does this read as English" pass CLAUDE.md Step 4 calls for, not a quantitative one.
The `axbench_llm_scores/` caches in this repo are a *different*, pre-existing artifact from the
original paper's own AxBench cross-validation (Gemma-2-9B-it) and are not part of this check.

## Per-feature, per-magnitude coherence

| feature | amp (+10) | sup (-2) | sup (-10) | sup (-20) |
|---|---|---|---|---|
| `2_13823` | degenerate (word salad, repeated "a"/"to") | **fluent** | degenerate (repeated "s", fragments) | degenerate (short garbled fragments) |
| `7_12166` | degenerate (repeated numbers/tokens) | **fluent** | degenerate (repeated capitalized words, no syntax) | degenerate (CJK/Arabic script noise, repeated glyphs) |
| `7_6944` | degenerate (kinship-word spam: "dad dad dad") | mostly fluent, some breakdown | degenerate (punctuation/bracket spam) | degenerate (repeated "that", bracket/unicode spam) |
| `14_6669` | borderline (short, thin, but grammatical) | **fluent** | degenerate (HTML-tag spam: `<strong><em><u>...`) | degenerate (repeated token "rature" + unicode noise) |

## Representative quotes

**`2_13823`, sup(-2)** — fluent:
> "I once heard that a person should have one job."
> "She saw a number of patients in her practice and she also does telemedicine appointments for which she has been to a"

**`2_13823`, sup(-10)** — degenerate:
> "Findings show thats."
> "She saw asfuls ands ins insfulls iss in; thes,\n\n and"

**`7_12166`, amp(+10)** — degenerate (this is the *amplification* condition, not suppression):
> "She saw a the the the 201 here <, <, <, <, <, <,"

**`14_6669`, sup(-2)** — fluent:
> "I believe that many people are familiar with the name Zhuge Liang."
> "She saw a little girl, maybe 10 years old, wearing a dress, walking across the street."

**`14_6669`, sup(-10)** — degenerate (markup-tag repetition, not language):
> "She saw a <strong><em><u><strong><em><u><sub><sub><strong><u> {{{ {{{ {{{ {{{ <sub><sub><sub> Simult"

**`14_6669`, sup(-20)** — degenerate (single-token repetition + non-English noise):
> "The news mentioned raturerature<strong> SDLK deportivaraturerature'rature' kuriosrature'rature'ratureratureraturerature"

## Cross-cutting finding: coherence tracks |s|, not sign

Coherence collapse is **not sign-specific**. Amplification at s=+10 is already degenerate for
3 of 4 features (`2_13823`, `7_12166`, `7_6944`) — these are not "well-behaved" features to begin
with at that magnitude. Suppression at s=-2 is fluent for all 4. Suppression at s=-10 and s=-20
both collapse into non-language output (repeated tokens, HTML/markup spam, foreign-script/unicode
noise) for all 4 features. The pattern across the sweep is: **fluent at small |s|, degenerate at
large |s|, in both directions** — a magnitude effect, consistent with the off-distribution
guardrail in CLAUDE.md, not evidence of a suppression-specific mechanism.

## Implication for the `14_6669` "monotonic reversal" claim

`signed_reanalysis.md` flagged `14_6669` as the strongest bidirectional-effect candidate because
its output_score rises monotonically with suppression magnitude (0.073 → 0.183 → 0.234 → 0.387 as
s goes 10 → -2 → -10 → -20). The generations show that rise coincides exactly with the text
degrading into repeated-token/markup spam at s=-10 and s=-20 — the same failure mode seen in the
other 3 (non-monotonic, noise-floor) candidates. This is the circularity Step 4 exists to catch:
a high output_score at extreme magnitude is consistent with the metric rewarding repetitive/
degenerate generation, not with genuine bidirectional semantic control. **Recommend downgrading
`14_6669` from "strongest reversal candidate" to "consistent with a degenerate-generation
artifact"** pending a quantitative check (e.g., repetition rate or perplexity at each magnitude,
which would let "high score + garbage text" be flagged automatically instead of by manual read).

## What remains

- No repetition-rate or perplexity metric was computed — this pass is qualitative only. Adding one
  would make the "degenerate at high |s|" observation reproducible/automatable across the full
  2600-feature set rather than just these 4 hand-picked candidates.
- Only the 4 features that survived `signed_reanalysis.md`'s noise-floor filter were read. This
  finding does not itself speak to the ~94.9% of features where amplify simply outscores suppress
  (no reversal to begin with).

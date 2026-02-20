# Paper Completion: Status Updates

## Update 3 (current)

**Completed:**
- Third revision pass on `paper_claude.md`
- Compiled paper successfully with pandoc

**What I did (this pass):**
- Added Grimmer/Roberts/Stewart observation to Discussion: text-as-data methodology reviews don't discuss calibration as a requirement for prevalence estimation, strengthening the "this gap is overlooked" argument
- Added practical LLM consideration to limitations: modern LLMs produce text not probabilities, so obtaining calibratable scores (via log-probs, confidence elicitation, repeated sampling) is a non-trivial step
- Added justification for synthetic shifts: measuring bias requires knowing true prevalence, which is unavailable under natural shifts; synthetic shifts are realistic in magnitude
- Total word count: ~3,255 (including markup), well under PNAS ~4,500 word limit

**Assessment of current state:**
- The paper is structurally complete and makes a clear, well-supported argument
- All TODOs from the original draft are resolved
- Key co-author feedback from llm_context.md has been addressed (X→Y assumption, bias vs variance, LLM hook, device terminology, Grimmer point)
- Two items from co-author notes were not included: (1) post-treatment variable analogy (doesn't fit cleanly in a short paper), (2) footnote about float output from LLMs (addressed instead in limitations paragraph)
- The incomplete `recalibrating2020` citation in references.bib is not cited in the paper so doesn't affect compilation

**Remaining items for the authors:**
- Review whether the title ("Multicalibration Is Necessary for Unbiased Model-Based Prevalence Estimation") is preferred over the original
- Fill in `[repository URL]` in Materials and Methods once repo is public
- Run the ACS notebook to generate `figure2_acs_age_shift.png` (requires internet access to download Census data)
- Consider whether additional analyses would strengthen the paper (sensitivity to subgroup specification, varying model quality)
- Complete the `recalibrating2020` citation in references.bib if it will be cited

## Update 2

**Completed:**
- Second revision pass with substantive additions

**What I did:**
- Added explicit X→Y causal assumption and covariate shift justification
- Added bias vs. variance paragraph explaining why bias is the central concern
- Added necessity argument for multicalibration
- Added paragraph about inadequacy of standard reporting practices (accuracy/F1/AUC)
- Strengthened transitions between sections
- Added references to Calibrate-Extrapolate framework and Halterman codebook LLMs

## Update 1

**Completed:**
- Read the full current draft, llm_context.md, and references.bib
- Wrote complete first draft of `paper_claude.md`

**What I did:**
- Restructured for PNAS format: Significance statement (120 words), Abstract (250 words), Introduction, Results, Discussion, Materials and Methods
- Filled all TODOs: abstract, significance statement, the "AI Systems as Measurement Devices" section (folded into intro), the connection between multicalibration and measurement, the discussion/conclusion
- Tightened the prose throughout—removed redundancies, made the argument flow more linearly
- Moved the simulation appendix detail into a brief Materials and Methods section (the full appendix can go in SI)
- Unified the theoretical argument: standard methods fail because they ensure only *marginal* calibration, multicalibration ensures *conditional* calibration, and the law of iterated expectations does the rest

**Key structural decisions:**
- Dropped the separate "AI Systems as Measurement Devices" section—the LLM context is now woven into the introduction (more concise)
- Combined the theory and simulation into a single Results section with three subsections
- The appendix (simulation details) is referenced as SI but not included in the main text to stay within ~4,500 words
- Kept both figures and the table

**What I plan to do next:**
- Re-read the draft critically and iterate on weak spots
- Check if any additional references are needed
- Verify word count is within PNAS limits
- Consider whether additional analysis would strengthen the paper (e.g., sensitivity analysis, more detailed OOD results)

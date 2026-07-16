# Formate, purines and energy metabolism: a glue-regulated model

Kinetic model of formate, adenine nucleotide and energy metabolism in
proliferating cells, extended with the PPAT–NUDT5 metabolite glue.

This repository contains the code, model outputs and manuscript for:

> A. Vazquez, *An allosteric metabolite glue couples AICAR inversely to
> one-carbon availability: a theoretical prediction and its therapeutic
> corollaries* (submitted).

The model extends that of Oizel et al., *Formate induces a metabolic switch in
nucleotide and energy metabolism*, Cell Death & Disease 11:310 (2020),
[doi:10.1038/s41419-020-2523-z](https://doi.org/10.1038/s41419-020-2523-z),
by replacing its demand-slaving assumption with an explicit PRPP pool and a
PPAT step gated by the PPAT–NUDT5 glue characterised in

- S.R. Witus et al., Nature (2026),
  [doi:10.1038/s41586-026-10790-3](https://doi.org/10.1038/s41586-026-10790-3)
- A. Strefeler et al., Nature Metabolism (2025),
  [doi:10.1038/s42255-025-01419-2](https://doi.org/10.1038/s42255-025-01419-2)

## Layout

```
switch.py            Published Oizel et al. (2020) model (baseline)
switch_ppat.py       Extended glue-regulated model; regenerates data/
data/                Model outputs underlying the figures
paper/
  formate_ppat_jtb.tex, .pdf   The manuscript (Elsevier CAS single-column)
  make_figs.py                 Regenerates the figures from switch_ppat.py
  fig1_calibration.pdf ...      Figures 1–4
  cas-sc.cls, cas-common.sty, cas-model2-names.bst
```

## Reproducing

Requires Python with numpy, pandas, scipy and matplotlib, and a LaTeX
distribution.

```
python switch_ppat.py       # regenerate data/*.csv (all points converge)
python paper/make_figs.py   # regenerate paper/fig1..4.pdf
cd paper && pdflatex formate_ppat_jtb.tex   # build the manuscript (run twice)
```

Every parameter in `switch_ppat.py` carries an inline provenance tag: `[PUB]`
(from the published model), `[FIT]` (fitted to Witus et al. binding data),
`[LIT]` (other literature) or `[ASM]` (assumed).

## Model

Eight coupled steady-state balances are solved for the free cytosolic
concentrations of AMP, ADP, ATP, formate, 10-formyl-THF, PRPP, GAR and AICAR.
The glue inhibits PPAT as a function of the competing AMP and PRPP levels,
calibrated entirely against published binding data. The central prediction is
that AICAR is coupled **inversely** to one-carbon availability, and that this
inverse coupling requires the glue — a NUDT5 knockout should reverse its sign.
See the manuscript for the equations, calibration and results.

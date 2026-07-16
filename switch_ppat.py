#!/usr/bin/env python3
"""
Formate / purine / energy metabolism with explicit PRPP and glue-regulated PPAT.

Extends the model of Oizel et al., Cell Death Dis 11:310 (2020)
[doi:10.1038/s41419-020-2523-z], implemented in switch.py, with:

  - an explicit PRPP pool and PRPP-dependent PPAT step,
  - AMP/PRPP-gated inhibition of PPAT by the PPAT-NUDT5 metabolite glue
    [Witus et al., Nature (2026), doi:10.1038/s41586-026-10790-3],
  - explicit CHO-THF, GAR and AICAR pools, so that the one-carbon-limited
    formyl transfer steps (GART, ATIC) can back up.

The published model slaves purine synthesis to demand (its Eq. 17). This
version replaces that assumption with a rate law for PPAT under the measured
glue feedback, and asks whether the resulting model reproduces the elevation
of GAR and AICAR at low one-carbon availability reported in Fig. 3a,b of the
2020 paper.

Parameter provenance is tagged in the comments:
  [PUB]  taken unchanged from switch.py / the 2020 supplementary text
  [FIT]  fitted here to published data of Witus et al. (2026)
  [ASM]  assumed; not constrained by either paper (see sensitivity analysis)
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, least_squares

# --------------------------------------------------------------------------
# Published parameters (Oizel et al. 2020)
# --------------------------------------------------------------------------

for0 = 0.8              # mM, intracellular formate [PMID:27211901]          [PUB]
ffor0 = 0.5             # mM/h, formate release, HAP1 WT [PMID:29636461]     [PUB]
kF = ffor0 / for0       # 1/h, formate exchange per unit formate             [PUB]

RNA = 20.0              # mM, RNA content [PMID:29044214]                    [PUB]
DNA = 10.0              # mM, DNA content [PMID:29044214]                    [PUB]
ADN = 5.0               # mM, free adenine pool                              [PUB]

mu_max = 1 / 24         # 1/h, 1 doubling/day                                [PUB]
epsilon = 6000.0        # mM ATP/doubling [PMID:29044214]                    [PUB]
m = 84.0                # mM/h, maintenance [PMID:29044214]                  [PUB]
e = 2 * (m + mu_max * epsilon)      # mM/h                                   [PUB]
a = e                               # mM/h, max ATPase rate                  [PUB]
eg = 0.75 * e                       # mM/h, max glycolytic rate              [PUB]
eo = 0.25 * e                       # mM/h, max OxPhos rate                  [PUB]
h = mu_max * (2 * ADN + RNA + 5 * DNA / 4)   # mM/h, FTHFS capacity          [PUB]

Eg = 0.25               # mM, glycolysis half-saturation for ADP             [PUB]
Eo = 0.12               # mM, OxPhos half-saturation for ADP                 [PUB]
H = 0.01                # mM, FTHFS half-saturation for formate              [PUB]
A = 2.0                 # mM, half-saturation of ATP-dependent growth        [PUB]
D = 0.2                 # mM, half-saturation of ADP-dependent growth        [PUB]
K = 0.8                 # ADK equilibrium [AMP][ATP]/[ADP]^2                 [PUB]

# Stoichiometry implied by the published 1C demand mu*(2*ADN + RNA + 5*DNA/4):
# 2 one-carbon units per purine (A and G), 1 per thymidylate. With RNA and DNA
# taken as 50% purines, this decomposes as
PUR_POOL = ADN + RNA / 2 + DNA / 2      # 20.0 mM purine equivalents
ADE_POOL = ADN + RNA / 4 + DNA / 4      # 12.5 mM adenine equivalents
TMP_POOL = DNA / 4                      # 2.5 mM thymidylate equivalents
PHI_ADE = ADE_POOL / PUR_POOL           # 0.625, adenine fraction of purine flux
# Consistency check against the published lumped constant:
assert abs((2 * PUR_POOL + TMP_POOL) * mu_max - h) < 1e-12

v_pur_max = mu_max * PUR_POOL           # 0.833 mM/h, purine demand at mu_max

# --------------------------------------------------------------------------
# Metabolite-glue calibration (Witus et al. 2026, Fig. 3b,c)
# --------------------------------------------------------------------------
#
# Mechanism reported: PPAT4-NUDT5_4 assembles via the NUDT5 R70-Y74 interface
# independently of nucleotide. PRPP binds the PPAT active site and dismantles
# the complex; AMP binds the same site, bridges PPAT to the NUDT5 C terminus,
# and shields the complex from PRPP-driven disassembly. AMP and PRPP therefore
# compete, and the inhibited fraction is
#
#     theta(AMP, PRPP) = fN * (1 + (AMP/Ka)^n) / (1 + (AMP/Ka)^n + PRPP/KdP)
#
# so that the PRPP concentration giving half-maximal dissociation is
#
#     IC50(AMP) = KdP * (1 + (AMP/Ka)^n)
#
# fN in [0,1] scales the maximum inhibited fraction with NUDT5 availability
# relative to PPAT (fN = 1 is saturating NUDT5; fN = 0 disables the feedback).

# Fig. 3c, wild-type NUDT5: PRPP IC50 for complex dissociation vs [AMP]
GLUE_AMP_MM = np.array([0.0, 0.1, 1.0])             # mM AMP
GLUE_IC50_UM = np.array([104.0, 327.0, 4900.0])     # uM PRPP
GLUE_IC50_SD = np.array([2.0, 19.0, 480.0])         # uM

KdP = GLUE_IC50_UM[0] / 1000.0          # mM, PRPP dissociation constant      [FIT]


def _ic50_resid(p):
    """log-residuals of IC50(AMP) = KdP*(1 + (AMP/Ka)^n) against Fig. 3c."""
    Ka, n = np.exp(p)
    pred = KdP * 1000.0 * (1 + (GLUE_AMP_MM / Ka) ** n)
    return np.log(pred[1:]) - np.log(GLUE_IC50_UM[1:])


_fit = least_squares(_ic50_resid, np.log([0.05, 1.3]))
Ka, n_hill = np.exp(_fit.x)             # mM, Hill coefficient                [FIT]


def glue_theta(amp, prpp, fN=1.0):
    """Fraction of PPAT held in the inhibited PPAT-NUDT5 complex."""
    x = (amp / Ka) ** n_hill
    return fN * (1.0 + x) / (1.0 + x + prpp / KdP)


def predicted_Ki_amp(prpp_mM=1.0, fN=1.0):
    """AMP giving half-maximal PPAT inhibition at fixed PRPP.

    Independent cross-check: Fig. 3b measured Ki = 274 +/- 33 uM (WT NUDT5)
    in an enzymatic assay at 1 mM PRPP, whereas Ka and n above were fitted to
    the AlphaLisa dissociation data of Fig. 3c.
    """
    lo, hi = glue_theta(0.0, prpp_mM, fN), glue_theta(1e6, prpp_mM, fN)
    target = 0.5 * (lo + hi)
    from scipy.optimize import brentq
    return brentq(lambda A: glue_theta(A, prpp_mM, fN) - target, 1e-6, 1e4)


# --------------------------------------------------------------------------
# Extended-model parameters
# --------------------------------------------------------------------------

Ftot = 0.02             # mM, total folate pool; the 2020 supplementary cites
                        # 0.004-2.5 mM [PMID:3497925] for H4PteGlu_n           [ASM]
Kc = 0.002              # mM, GART/ATIC half-saturation for CHO-THF            [ASM]
Kg = 0.01               # mM, GART half-saturation for GAR                     [ASM]
Kt = 0.01               # mM, ATIC half-saturation for AICAR                   [ASM]
c_gart = 5.0            # GART/ATIC capacity, x purine demand at mu_max        [ASM]
gmax = c_gart * v_pur_max
tmax = c_gart * v_pur_max

Kprpp = 0.3             # mM, PPAT half-saturation for PRPP                    [ASM]
c_ppat = 2.0            # PPAT capacity, x purine demand at mu_max             [ASM]
Pmax = c_ppat * v_pur_max

c_prps = 2.0            # PRPP supply, x purine demand at mu_max               [ASM]
S_prpp = c_prps * v_pur_max
ko_prpp = 2.0           # 1/h, lumped PRPP drain (salvage, pyrimidine, NAD)    [ASM]

kdeg = 0.1              # 1/h, drain of GAR/AICAR (dephosphorylation/export;
                        # AICAR is excreted as AICA riboside)                  [ASM]

# Reversible FTHFS. The published model treats
#     formate + THF + ATP -> 10-CHO-THF + ADP + Pi
# as irreversible. It is not: MTHFD1 is trifunctional and its synthetase domain
# runs both ways, which is the only route by which cytosolic one-carbon units
# (SHMT1 -> CH2-THF -> CHO-THF, no free formate intermediate) can reach the
# formate pool at all. Keq is dimensionless,
#     Keq = [CHO-THF][ADP][Pi] / ([formate][THF][ATP])
Keq_fthfs = 40.0        # 20-61 [Himes & Rabinowitz, JBC 1962]                 [LIT]
Pi = 5.0                # mM, intracellular orthophosphate                     [ASM]

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

STATE = ['amp', 'adp', 'atp', 'form', 'cho', 'prpp', 'gar', 'aic']


def _unpack(x):
    """Map unconstrained reals to positive concentrations.

    CHO-THF is bounded by the total folate pool, so it is mapped through a
    logistic; everything else is a log transform. This enforces positivity
    structurally and avoids the negative-concentration roots that fsolve
    returns for the published constant-mu variant.
    """
    z = np.clip(np.asarray(x, dtype=float), -200.0, 200.0)
    amp, adp, atp, form, prpp, gar, aic = np.exp(z[[0, 1, 2, 3, 5, 6, 7]])
    cho = Ftot / (1.0 + np.exp(-z[4]))
    return amp, adp, atp, form, cho, prpp, gar, aic


def fluxes(x, fSCFr, form_x, fSC10THF_cytosol, growth_rate_model='atp_dependent',
           fN=1.0, cytosolic_via_formate=False, fthfs_reversible=False):
    """All rates and derived quantities for a given state."""
    amp, adp, atp, form, cho, prpp, gar, aic = _unpack(x)

    if growth_rate_model == 'atp_dependent':
        mu = (-m + a * atp / (A + atp)) / epsilon
    elif growth_rate_model == 'adp_dependent':
        mu = (-m + a * adp / (D + adp)) / epsilon
    else:
        raise ValueError(growth_rate_model)

    # One-carbon inputs. The 2020 supplementary (Eq. 16) routes cytosolic
    # serine catabolism straight to CHO-THF; switch.py line 51 instead adds it
    # to the formate balance. cytosolic_via_formate reproduces the latter.
    f_ser_formate = h * fSCFr                       # mitochondrial serine -> formate
    f_ser_cho = h * fSC10THF_cytosol                # cytosolic serine -> CHO-THF
    if cytosolic_via_formate:
        f_ser_formate, f_ser_cho = f_ser_formate + f_ser_cho, 0.0

    thf = Ftot - cho
    if fthfs_reversible:
        # Irreversible Michaelis-Menten scaled by the thermodynamic driving
        # force (1 - Gamma/Keq). Written expanded rather than as an explicit
        # (1 - Gamma/Keq) factor: Gamma diverges as form -> 0 while the product
        # stays finite, so the expanded form has no singularity. Reduces exactly
        # to the published irreversible law when Gamma << Keq and thf -> Ftot.
        v_fthfs = h / (Ftot * (H + form)) * (
            form * thf - cho * adp * Pi / (Keq_fthfs * atp))
    else:
        v_fthfs = h * (form / (H + form)) * (thf / Ftot)
    onec = cho / (Kc + cho)

    theta = glue_theta(amp, prpp, fN)
    v_ppat = Pmax * (prpp / (Kprpp + prpp)) * (1.0 - theta)
    v_gart = gmax * (gar / (Kg + gar)) * onec
    v_atic = tmax * (aic / (Kt + aic)) * onec
    v_tyms = mu * TMP_POOL

    ffor = kF * form - kF * form_x
    flac = eg * adp / (Eg + adp)
    fo = eo * adp / (Eo + adp)
    energy_charge = (atp + adp / 2) / (atp + adp + amp)

    return dict(amp=amp, adp=adp, atp=atp, form=form, cho=cho, prpp=prpp,
                gar=gar, aic=aic, mu=mu, theta=theta, v_ppat=v_ppat,
                v_gart=v_gart, v_atic=v_atic, v_tyms=v_tyms, v_fthfs=v_fthfs,
                f_ser_formate=f_ser_formate, f_ser_cho=f_ser_cho,
                ffor=ffor, flac=flac, fo=fo, energy_charge=energy_charge)


def residuals(x, fSCFr, form_x, fSC10THF_cytosol, growth_rate_model='atp_dependent',
              fN=1.0, cytosolic_via_formate=False, fthfs_reversible=False):
    r = fluxes(x, fSCFr, form_x, fSC10THF_cytosol, growth_rate_model, fN,
               cytosolic_via_formate, fthfs_reversible)
    mu = r['mu']
    y = np.zeros(8)
    # 1. formate balance
    y[0] = r['f_ser_formate'] + kF * form_x - r['v_fthfs'] - kF * r['form']
    # 2. CHO-THF balance
    y[1] = r['v_fthfs'] + r['f_ser_cho'] - r['v_gart'] - r['v_atic'] - r['v_tyms']
    # 3. PRPP balance
    y[2] = S_prpp - r['v_ppat'] - ko_prpp * r['prpp'] - mu * r['prpp']
    # 4. GAR balance
    y[3] = r['v_ppat'] - r['v_gart'] - (mu + kdeg) * r['gar']
    # 5. AICAR balance
    y[4] = r['v_gart'] - r['v_atic'] - (mu + kdeg) * r['aic']
    # 6. adenine balance
    y[5] = PHI_ADE * r['v_atic'] - mu * (r['amp'] + r['adp'] + r['atp'] + ADE_POOL)
    # 7. energy balance
    y[6] = a * r['atp'] / (A + r['atp']) - eg * r['adp'] / (Eg + r['adp']) \
        - eo * r['adp'] / (Eo + r['adp'])
    # 8. adenylate kinase equilibrium
    y[7] = r['amp'] * r['atp'] - K * r['adp'] ** 2
    return y


_X0_DEFAULT = np.array([np.log(0.01), np.log(0.1), np.log(1.0), np.log(0.05),
                        0.0, np.log(0.05), np.log(0.01), np.log(0.01)])
TOL = 1e-9


def solve(fSCFr, form_x, fSC10THF_cytosol, growth_rate_model='atp_dependent',
          fN=1.0, cytosolic_via_formate=False, fthfs_reversible=False,
          x0=None, seed=0):
    """Solve the steady state.

    Uses a trust-region least-squares solve rather than fsolve: the residuals
    span several orders of magnitude and fsolve stalls on them. Restarts from
    perturbed initial guesses until the residual norm is below tolerance.
    Returns (fluxes dict, state vector).
    """
    args = (fSCFr, form_x, fSC10THF_cytosol, growth_rate_model, fN,
            cytosolic_via_formate, fthfs_reversible)
    starts = [x0] if x0 is not None else []
    starts.append(_X0_DEFAULT)
    rng = np.random.default_rng(seed)
    starts += [_X0_DEFAULT + rng.normal(0, 2.0, 8) for _ in range(12)]

    best, best_r = None, np.inf
    for s in starts:
        try:
            sol = least_squares(residuals, s, args=args, xtol=1e-15,
                                ftol=1e-15, gtol=1e-15, max_nfev=5000)
        except (ValueError, FloatingPointError):
            continue
        rmax = np.max(np.abs(residuals(sol.x, *args)))
        if rmax < best_r:
            best, best_r = sol.x, rmax
        if best_r < TOL:
            break

    r = fluxes(best, *args)
    r['converged'] = bool(best_r < TOL)
    r['max_residual'] = float(best_r)
    return r, best


def scan(form_x, fSC10THF_cytosol, growth_rate_model='atp_dependent', fN=1.0,
         n=100, cytosolic_via_formate=False, fthfs_reversible=False):
    """Scan the rate of serine catabolism to formate, as in the 2020 paper.

    Sweeps forwards then backwards with continuation and keeps the better root
    at each point. A forward/backward disagreement would indicate hysteresis;
    see check_hysteresis().
    """
    grid = [2 * i / (n - 1) for i in range(n)]

    def sweep(order):
        out, x0 = {}, None
        for i in order:
            r, x = solve(grid[i], form_x, fSC10THF_cytosol, growth_rate_model,
                         fN, cytosolic_via_formate, fthfs_reversible, x0)
            if r['converged']:
                x0 = x
            out[i] = (r, x)
        return out

    fwd = sweep(range(n))
    bwd = sweep(range(n - 1, -1, -1))
    rows = []
    for i in range(n):
        rf, rb = fwd[i][0], bwd[i][0]
        r = rf if rf['max_residual'] <= rb['max_residual'] else rb
        rows.append({'fSCFr_pct': 100 * grid[i],
                     'atp_fwd': fwd[i][0]['atp'], 'atp_bwd': bwd[i][0]['atp'],
                     **{k: v for k, v in r.items()}})
    df = pd.DataFrame(rows)
    # one-carbon availability index of the 2020 paper: total 1C input, relative
    # to the FTHFS capacity h
    df['onec_index'] = (df['f_ser_formate'] + df['f_ser_cho'] + kF * form_x) / h
    return df


def check_hysteresis(df):
    """Max relative disagreement between the forward and backward sweeps."""
    return float(np.max(np.abs(df.atp_fwd - df.atp_bwd) / (df.atp_bwd.abs() + 1e-12)))


def energy_charge_scan(supply_factors=None, fSCFr=2.0, form_x=0.02,
                       fSC10THF_cytosol=0.0, fN=1.0):
    """Scan the adenylate energy charge by scaling the ATP-supply capacity.

    Free AMP is fixed by the adenylate kinase equilibrium, so it is low at high
    energy charge and rises steeply as energy charge falls. This scan reduces the
    glycolytic and oxidative-phosphorylation capacities (eg, eo) together by a
    common factor, at fixed one-carbon supply, and records where free AMP crosses
    the glue affinity Ka and engages the AMP-sensing arm of the feedback.
    """
    global eg, eo
    if supply_factors is None:
        supply_factors = np.linspace(1.0, 0.2, 25)
    eg0, eo0 = eg, eo
    rows = []
    for sup in supply_factors:
        eg, eo = sup * eg0, sup * eo0
        try:
            r, _ = solve(fSCFr, form_x, fSC10THF_cytosol, fN=fN)
            rows.append({'supply': sup, 'amp': r['amp'], 'adp': r['adp'],
                         'atp': r['atp'], 'energy_charge': r['energy_charge'],
                         'amp_over_Ka': r['amp'] / Ka, 'theta': r['theta'],
                         'mu_per_day': 24 * r['mu'], 'converged': r['converged']})
        except Exception:                                   # pragma: no cover
            rows.append({'supply': sup, 'converged': False})
    eg, eo = eg0, eo0
    return pd.DataFrame(rows)


def sensitivity(param_names=None, factor=2.0):
    """Vary each assumed parameter up and down and report the effect.

    Reports the span of the glue occupancy theta across the scan (does the
    feedback modulate at all?) and the GAR fold change between the lowest and
    highest one-carbon availability (the GAR overflow prediction).
    """
    import switch_ppat as M
    if param_names is None:
        param_names = ['Ftot', 'Kc', 'Kg', 'Kt', 'c_gart', 'Kprpp', 'c_ppat',
                       'c_prps', 'ko_prpp', 'kdeg']
    rows = []
    for name in param_names:
        base = getattr(M, name)
        for lab, val in [('down', base / factor), ('up', base * factor)]:
            setattr(M, name, val)
            M.gmax = M.tmax = M.c_gart * M.v_pur_max
            M.Pmax = M.c_ppat * M.v_pur_max
            M.S_prpp = M.c_prps * M.v_pur_max
            try:
                df = M.scan(0.02, 0.0, fN=1.0, n=25)
                rows.append({'param': name, 'dir': lab, 'value': val,
                             'converged': bool(df.converged.all()),
                             'theta_min': df.theta.min(), 'theta_max': df.theta.max(),
                             'theta_span': df.theta.max() - df.theta.min(),
                             'gar_fold': df.gar.iloc[0] / max(df.gar.iloc[-1], 1e-12),
                             'atp_max': df.atp.max()})
            except Exception as exc:                      # pragma: no cover
                rows.append({'param': name, 'dir': lab, 'value': val,
                             'converged': False, 'theta_span': np.nan,
                             'gar_fold': np.nan, 'atp_max': np.nan})
            setattr(M, name, base)
        M.gmax = M.tmax = M.c_gart * M.v_pur_max
        M.Pmax = M.c_ppat * M.v_pur_max
        M.S_prpp = M.c_prps * M.v_pur_max
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # Regenerate the model outputs that underlie the figures, into the
    # repository data/ directory, matching the Data availability statement in
    # paper/formate_ppat_jtb.tex. Figures are produced by paper/make_figs.py.
    import os
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(outdir, exist_ok=True)

    print(f'glue calibration [FIT]: Ka = {Ka*1000:.1f} uM, n = {n_hill:.2f}, '
          f'KdP = {KdP*1000:.0f} uM')
    Ki_pred = predicted_Ki_amp(1.0) * 1000
    print(f'  cross-check: predicted Ki(AMP) at 1 mM PRPP = {Ki_pred:.0f} uM '
          f'vs measured 274 +/- 33 uM (Witus et al. Fig. 3b)')

    conditions = [
        ('for0.02_cyt0.0', 0.02, 0.0),
        ('for1.0_cyt0.0', 1.0, 0.0),
        ('for0.02_cyt0.5', 0.02, 0.5),
    ]
    scans = {}
    for tag, fx, cyt in conditions:
        for fN, gl in [(1.0, 'glue_on'), (0.0, 'glue_off')]:
            df = scan(fx, cyt, fN=fN)
            scans[(tag, gl)] = df
            df.to_csv(os.path.join(outdir, f'formate.v6.{tag}.{gl}.csv'), index=False)
            print(f'formate.v6.{tag}.{gl}.csv: {df.converged.sum()}/{len(df)} '
                  f'converged, max residual {df.max_residual.max():.2e}, '
                  f'fwd/bwd gap {check_hysteresis(df):.1e}')

    d0 = scans[('for0.02_cyt0.0', 'glue_on')]
    print(f'\nglue occupancy theta across the scan: '
          f'{d0.theta.min()*100:.1f}% -> {d0.theta.max()*100:.1f}% '
          f'(span {100*(d0.theta.max()-d0.theta.min()):.1f} points)')
    print(f'model AMP range {d0.amp.min()*1000:.1f}-{d0.amp.max()*1000:.1f} uM '
          f'= {d0.amp.min()/Ka:.2f}-{d0.amp.max()/Ka:.2f} x Ka')

    print('\nsensitivity to assumed parameters (x2 up/down):')
    sens = sensitivity()
    sens.to_csv(os.path.join(outdir, 'sensitivity.v6.csv'), index=False)
    print(sens.to_string(index=False, float_format=lambda v: f'{v:.3g}'))

    print('\nenergy-charge scan (free AMP vs energy charge; AMP arm engagement):')
    ec = energy_charge_scan()
    ec.to_csv(os.path.join(outdir, 'energy_charge.v6.csv'), index=False)
    hi, lo = ec.iloc[0], ec[ec.energy_charge < 0.72].iloc[0]
    print(f'  energy charge {hi.energy_charge:.2f}: free AMP {hi.amp*1000:.0f} uM '
          f'({hi.amp_over_Ka:.2f} x Ka), theta {100*hi.theta:.0f}%')
    print(f'  energy charge {lo.energy_charge:.2f}: free AMP {lo.amp*1000:.0f} uM '
          f'({lo.amp_over_Ka:.2f} x Ka), theta {100*lo.theta:.0f}%')

    # the cyan (cytosolic-serine) condition, three one-carbon routings
    print('\ncyan condition (for_x=0.02, fS,CHO-THF=0.5), three routings:')
    ways = [
        ('code', dict(cytosolic_via_formate=True, fthfs_reversible=False)),
        ('supp', dict(cytosolic_via_formate=False, fthfs_reversible=False)),
        ('rev', dict(cytosolic_via_formate=False, fthfs_reversible=True)),
    ]
    for nm, kw in ways:
        df = scan(0.02, 0.5, fN=1.0, **kw)
        df.to_csv(os.path.join(outdir, f'formate.v6.cyan.{nm}.csv'), index=False)
        s = np.sign(df.ffor.values)
        i = np.where(np.diff(s) > 0)[0]
        thr = df.fSCFr_pct.values[i[0]] if len(i) else np.nan
        print(f'  {nm:5s}: [For] {df.form.min():.5f}-{df.form.max():.4f} mM, '
              f'overflow at fSCFr={thr:.0f}%, converged={df.converged.all()}')

    print('\nfigures: run paper/make_figs.py')

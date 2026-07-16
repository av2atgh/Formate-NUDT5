#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.optimize import fsolve


def f(adn, fSCFr, fgr, form_x, fSC10THF_cytosol, growth_rate_model):
    """
    Input parameters:
        fSCFr               relative, rate of serine catabolism to formate, relative to the FTHFS rate
        form_x              mM, extracellular formate concentration
        fSC10THF_cytosol    relative, rate of serine catabolism to 10-formyl-tetrahydrofolate from the cytosolic pathway, relative to the FTHFS rate
    """

    amp = adn[0]
    adp = adn[1]
    atp = adn[2]

    # Formate release
    for0 = 0.8          # mM, estimated intracellular formate concentration [PMID:27211901]
    ffor0 = 0.5         # mM/h Measurement for HAP1 WT cells [PMID:29636461]
    kF = ffor0 / for0   # estimated rate of formate exchange per unit of formate [Calculated]

    # Nucleotide concentration
    RNA = 20            # mM, RNA content [PMID:29044214, Table 1]
    DNA = 10            # mM, DNA content [PMID:29044214, Table 1]
    ADN = 5             # mM, free adenine pool [Assumed]

    # Maximum proliferation rate
    mu_max = 1 / 24     # 1 doubling/day [Assumed]
    epsilon = 6000      # mM ATP/doubling, energy demand to double the cell content [PMID:29044214]
    m = 84              # mM/h, energy demand of cell maintenance [PMID:29044214]
    e = 2 * (m + mu_max * epsilon)  # mM/h, assuming that the maximum energy production rate matches the maximum energy demand
    a = e               # mM/h, assuming that the maximum rate of the growth limiting step matches the maximum energy production rate
    eg = 0.75 * e * fgr
    eo = 0.25 * e
    h = mu_max * (2 * ADN + RNA + 5 * DNA / 4)  # mM/h, assuming that the FTHFS rate matches the 1C demand of purine synthesis at maximum proliferation rate

    # Saturation constants
    Eg = 0.25           # mM, Fit to glycolysis model [This work]
    Eo = 0.12           # Fit to OxPhos model [This work]
    H = 0.01            # 0.004-2.5 mM [PMID:3497925], Fig. 3, H4PteGlu_n, n=3-5
    A = 2               # mM, half saturation of ATP dependent growth rate [Assumed]
    D = 0.2             # mM, half saturation of ADP dependent growth rate [Assumed]

    # Adenylate kinase
    K = 0.8             # Equilibrium constant, [AMP][ATP]/[ADP]^2 [5924638,198793]

    # Total 1C production
    fSCF = kF * form_x + h * fSCFr + h * fSC10THF_cytosol

    # Growth rate model
    if growth_rate_model == 'adp_dependent':
        mu = (-m + a * adp / (D + adp)) / epsilon
    elif growth_rate_model == 'atp_dependent':
        mu = (-m + a * atp / (A + atp)) / epsilon
    elif growth_rate_model == 'constant':
        mu = mu_max
    else:
        raise ValueError(f"Unknown growth_rate_model: {growth_rate_model}")

    form = (((fSCF - h) / kF - H + np.sqrt(((fSCF - h) / kF - H) ** 2 + 4 * fSCF * H / kF)) / 2)

    ffor = kF * form - kF * form_x

    flac = eg * adp / (Eg + adp)

    fo = eo * adp / (Eo + adp)

    energy_charge = (atp + adp / 2) / (atp + adp + amp)

    # Balance equations
    y = np.zeros(3)
    y[0] = a * atp / (A + atp) - eg * adp / (Eg + adp) - eo * adp / (Eo + adp)
    y[1] = amp * atp - K * adp * adp
    y[2] = h * form / (H + form) - mu * (2 * amp + 2 * adp + 2 * atp + RNA + 5 * DNA / 4)  # 2 1C units per A and G, 1 1C unit per T

    return y, mu, form, ffor, flac, fo, energy_charge


def f_residuals(adn, fSCFr, fgr, form_x, fSC10THF_cytosol, growth_rate_model):
    """Wrapper returning only residuals for fsolve."""
    y, *_ = f(adn, fSCFr, fgr, form_x, fSC10THF_cytosol, growth_rate_model)
    return y


### Simulation of increasing serine catabolism to formate

n = 100
nr = 2
nr1 = 2

for_x_min = 0.02    # 1mM, minimum simulated extracellular formate concentration
for_x_max = 1       # 1mM, maximum simulated extracellular formate concentration

fSC10THF_cytosol_min = 0    # minimum simulated rate of serine catabolism to 10-formyl-tetrahydrofolate from the cytosolic pathway
fSC10THF_cytosol_max = 0.5  # maximum simulated rate of serine catabolism to 10-formyl-tetrahydrofolate from the cytosolic pathway

growth_rate_models = {1: 'adp_dependent', 2: 'atp_dependent', 3: 'constant'}

for r in range(1, nr + 1):     # loop over different concentrations of extracellular formate

    for_x = for_x_min + (r - 1) * (for_x_max - for_x_min) / (nr - 1)

    for r1 in range(1, nr1 + 1):    # loop over different rates of serine catabolism to 10-formyl-tetrahydrofolate from the cytosolic pathway

        fSC10THF_cytosol = fSC10THF_cytosol_min + (r1 - 1) * (fSC10THF_cytosol_max - fSC10THF_cytosol_min) / (nr1 - 1)

        for g in [1, 2, 3]:

            growth_rate_model = growth_rate_models[g]

            filename = f'formate.v5.for_{for_x:.0f}.fSC10THF_cytosol_{fSC10THF_cytosol:.1f}.{growth_rate_model}.csv'
            print(filename)

            rows = []
            for i in range(1, n + 1):   # loop over different rates of serine catabolism to formate
                fSCFr = 2 * (i - 1) / (n - 1)
                adn0 = np.array([0.01, 0.1, 1.0])   # AMP, ADP, ATP in mM

                adn = fsolve(f_residuals, adn0, args=(fSCFr, 1, for_x, fSC10THF_cytosol, growth_rate_model))
                y, mu, form, ffor, flac, fo, energy_charge = f(adn, fSCFr, 1, for_x, fSC10THF_cytosol, growth_rate_model)

                rows.append({
                    'fSCFr_pct': 100 * fSCFr,
                    'amp': adn[0],
                    'adp': adn[1],
                    'atp': adn[2],
                    'form': form,
                    'mu_per_day': 24 * mu,
                    'ffor': ffor,
                    'flac': flac,
                    'fo_norm': fo / 5,
                })

            pd.DataFrame(rows).to_csv(filename, index=False)

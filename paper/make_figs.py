#!/usr/bin/env python3
"""Figures for the J. Theor. Biol. submission. Writes PDFs into this directory."""
import sys, os
import numpy as np, pandas as pd
# switch_ppat.py lives one directory up (repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import switch_ppat as M  # noqa: E402
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 200,
                     'font.family': 'serif'})

N = 60
on = M.scan(0.02, 0.0, fN=1.0, n=N)      # wild type: glue engaged
ko = M.scan(0.02, 0.0, fN=0.0, n=N)      # NUDT5 KO: glue removed, PPAT intact
xk = on.onec_index

# ---------------------------------------------------------------- Figure 1
# glue calibration + the AMP window
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
A = np.logspace(-3, 1, 200)
ax[0].errorbar(M.GLUE_AMP_MM, M.GLUE_IC50_UM, yerr=M.GLUE_IC50_SD, fmt='o',
               color='k', ms=4, capsize=2, label='Witus et al. (2026), Fig. 3c')
ax[0].plot(A, M.KdP*1000*(1+(A/M.Ka)**M.n_hill), '-', color='crimson',
           label=f'fit: $K_a$={M.Ka*1000:.0f} µM, $n$={M.n_hill:.2f}')
ax[0].set(xscale='log', yscale='log', xlabel='[AMP] (mM)',
          ylabel=r'PRPP IC$_{50}$ (µM)', xlim=(0.03, 3))
ax[0].legend(frameon=False, fontsize=6, loc='upper left')
ax[0].set_title('a', loc='left', fontweight='bold')

Aum = np.logspace(-1, 4, 300)
for prpp, c in [(0.1, '#1f77b4'), (0.44, 'crimson'), (1.0, '#2ca02c')]:
    ax[1].plot(Aum, M.glue_theta(Aum/1000, prpp), color=c,
               label=f'[PRPP]={prpp*1000:.0f} µM')
lo, hi = on.amp.min()*1000, on.amp.max()*1000
ax[1].axvspan(lo, hi, color='gold', alpha=.35, lw=0)
ax[1].text(np.sqrt(lo*hi), 0.42, 'model\nAMP', ha='center', fontsize=6, color='#8a6d00')
ax[1].axvline(M.Ka*1000, ls=':', color='k', lw=.8)
ax[1].text(M.Ka*1000*1.2, 0.03, '$K_a$', fontsize=6)
ax[1].set(xscale='log', xlabel='[AMP] (µM)', ylabel=r'$\theta$ (PPAT inhibited)', ylim=(0, 1))
ax[1].legend(frameon=False, fontsize=6, loc='upper left')
ax[1].set_title('b', loc='left', fontweight='bold')
fig.tight_layout(); fig.savefig('fig1_calibration.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Figure 2
# THE KEY FINDING: AICAR vs one-carbon availability, WT vs KO
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
ax[0].plot(xk, on.aic, '-', color='crimson', lw=1.6, label='wild type (glue on)')
ax[0].plot(xk, ko.aic, '--', color='#555', lw=1.6, label=r'$\Delta$NUDT5 (glue off)')
ax[0].set(xlabel='one-carbon availability index', ylabel='[AICAR] (mM)', yscale='log')
ax[0].legend(frameon=False, fontsize=6.5, loc='lower left')
ax[0].set_title('a', loc='left', fontweight='bold')
ax[0].annotate('WT: AICAR falls\nwith 1C', (xk.iloc[-1], on.aic.iloc[-1]),
               (xk.iloc[-1]*0.62, on.aic.iloc[-1]*3.2), fontsize=6, color='crimson',
               arrowprops=dict(arrowstyle='->', color='crimson', lw=.7))

ax[1].plot(xk, on.v_ppat, '-', color='crimson', lw=1.4, label=r'$v_{\rm PPAT}$, WT')
ax[1].plot(xk, ko.v_ppat, '--', color='crimson', lw=1.0, label=r'$v_{\rm PPAT}$, $\Delta$NUDT5')
ax[1].plot(xk, on.v_atic, '-', color='#1f77b4', lw=1.4, label=r'$v_{\rm ATIC}$ (demand)')
ax[1].set(xlabel='one-carbon availability index', ylabel='flux (mM h$^{-1}$)')
ax[1].legend(frameon=False, fontsize=6, loc='lower right')
ax[1].set_title('b', loc='left', fontweight='bold')
fig.tight_layout(); fig.savefig('fig2_aicar_prediction.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Figure 3
# scan overview: formate, ATP, AMP, PRPP, theta, mu
fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.0))
panels = [('form', '[formate] (mM)', True), ('atp', '[ATP] (mM)', False),
          ('prpp', '[PRPP] (mM)', False), ('theta', r'$\theta$', False),
          ('gar', '[GAR] (mM)', True), ('mu', r'$\mu$ (h$^{-1}$)', False)]
for axi, (col, lab, logy) in zip(axes.ravel(), panels):
    axi.plot(on.fSCFr_pct, on[col], '-', color='crimson', lw=1.3, label='WT')
    axi.plot(ko.fSCFr_pct, ko[col], '--', color='#555', lw=1.3, label=r'$\Delta$NUDT5')
    axi.set(xlabel='serine catabolism to formate (%)', ylabel=lab)
    if logy: axi.set_yscale('log')
    if col == 'theta': axi.set_ylim(0, 1)
axes.ravel()[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig('fig3_scan.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Figure 4
# free AMP and glue occupancy versus adenylate energy charge
ec = M.energy_charge_scan()
ec = ec[ec.converged].sort_values('energy_charge')
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
ax[0].plot(ec.energy_charge, ec.amp * 1000, '-', color='crimson', lw=1.6)
ax[0].axhline(M.Ka * 1000, ls=':', color='k', lw=.9)
ax[0].text(ec.energy_charge.min(), M.Ka * 1000 * 1.15, '$K_a$', fontsize=7)
ax[0].axvspan(0.90, 0.96, color='gold', alpha=.35, lw=0)
ax[0].text(0.93, ec.amp.min() * 1000 * 2, 'proliferating', ha='center',
           fontsize=6, color='#8a6d00', rotation=90)
ax[0].set(xlabel='adenylate energy charge', ylabel='free [AMP] (µM)', yscale='log')
ax[0].invert_xaxis()
ax[0].set_title('a', loc='left', fontweight='bold')
ax[1].plot(ec.energy_charge, 100 * ec.theta, '-', color='#1f77b4', lw=1.6)
ax[1].axvspan(0.90, 0.96, color='gold', alpha=.35, lw=0)
ax[1].set(xlabel='adenylate energy charge', ylabel=r'$\theta$ (PPAT inhibited, %)',
          ylim=(0, 100))
ax[1].invert_xaxis()
ax[1].set_title('b', loc='left', fontweight='bold')
fig.tight_layout(); fig.savefig('fig4_energy_charge.pdf'); plt.close(fig)

print('wrote fig1_calibration.pdf fig2_aicar_prediction.pdf fig3_scan.pdf '
      'fig4_energy_charge.pdf')
print(f'AICAR WT: {on.aic.iloc[0]:.3f} -> {on.aic.max():.3f} -> {on.aic.iloc[-1]:.3f}')
print(f'AICAR KO: {ko.aic.iloc[0]:.3f} -> {ko.aic.max():.3f} -> {ko.aic.iloc[-1]:.3f}')

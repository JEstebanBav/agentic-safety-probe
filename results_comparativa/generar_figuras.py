# -*- coding: utf-8 -*-
"""Figuras comparativas entre modelos para la presentacion de tesis."""
import json, io, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = r"C:/Users/38614180/AgentHarm/agentic-safety-probe"
OUT = os.path.join(P, 'results_comparativa', 'figures')
os.makedirs(OUT, exist_ok=True)

COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
GRID = '#d6d5d0'
INK = '#0b0b0b'
INK2 = '#52514e'

MOD = [
    ('results_llama33',           'Llama-3.3-70B\nFP16', 80, 52),
    ('results_qwen32b',           'Qwen3-32B\n4-bit',    64, 46),
    ('results_qwen8b',            'Qwen3-8B\nFP16',      36, 22),
    ('results_mistral7b_4bit',    'Mistral-7B\n4-bit',   32, 16),
    ('results_mistral7b_fp16_v2', 'Mistral-7B\nFP16',    32, 16),
]

D = []
for d, nom, NL, L in MOD:
    o = json.load(io.open(os.path.join(P, d, 'results.json'), encoding='utf-8'))
    fe = o['format_effect']['per_layer'][str(L)]
    adl = (o.get('agent_direction') or {}).get('layer_results', {})
    cos = dict((k, v['cosine_similarity']) for k, v in adl.items() if isinstance(v, dict))
    D.append(dict(nom=nom, NL=NL, L=L, cos=cos,
                  dph=fe['delta_p_harmful'],
                  interd=fe['interaction_cohens_d'],
                  kappa=fe['scale_contraction'],
                  aurc=fe['auroc_chat'], aura=fe['auroc_agent']))

plt.rcParams.update({'font.size': 13, 'axes.edgecolor': '#9a998f',
                     'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK2, 'ytick.color': INK2,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def style(ax):
    ax.grid(axis='y', color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def es(v, n=2):
    return (('%.' + str(n) + 'f') % v).replace('.', ',')


names = [d['nom'] for d in D]
x = np.arange(len(D))

# ---------- FIGURA A: crudo vs estandarizado ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.4))
v = [d['dph'] for d in D]
a1.bar(x, v, color=COL, width=0.62)
a1.set_yscale('log')
style(a1)
a1.set_xticks(x)
a1.set_xticklabels(names, fontsize=11)
a1.set_ylabel(r'$\Delta p_h$  (unidades crudas, escala log)')
a1.set_title('Unidades crudas:\nno comparables entre modelos', fontsize=14, pad=12)
for i, val in enumerate(v):
    a1.text(i, val * 1.14, es(val), ha='center', fontsize=12.5, color=INK)
a1.set_ylim(1, 260)
a1.text(0.5, 0.035, 'rango de 45x entre el menor y el mayor', transform=a1.transAxes,
        ha='center', fontsize=11.5, color=INK2, style='italic')

v2 = [d['interd'] for d in D]
a2.bar(x, v2, color=COL, width=0.62)
style(a2)
a2.set_xticks(x)
a2.set_xticklabels(names, fontsize=11)
a2.set_ylabel(r'$d$ de Cohen de la interacción $\Delta^2 p$')
a2.set_title('Unidades estandarizadas:\nel mismo efecto en todos los modelos',
             fontsize=14, pad=12)
a2.axhline(0.8, color='#52514e', linestyle='--', linewidth=1.4)
a2.text(len(D) - 0.45, 0.9, 'd = 0,8  (efecto grande)', ha='right', fontsize=11, color=INK2)
for i, val in enumerate(v2):
    a2.text(i, val + 0.08, es(val), ha='center', fontsize=12.5, color=INK)
a2.set_ylim(0, 3.6)
fig.suptitle('El mismo efecto medido en dos unidades, en tres familias de arquitectura',
             fontsize=16.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, 'cmp_crudo_vs_estandarizado.png'), dpi=150)
fig.savefig(os.path.join(OUT, 'cmp_crudo_vs_estandarizado.pdf'))
plt.close(fig)

# ---------- FIGURA B: rotacion ----------
fig, ax = plt.subplots(figsize=(13.5, 7.0))
ax.axhspan(0, 0.15, color='#f0efec', zorder=0)
for i, d in enumerate(D):
    ls = sorted(d['cos'].keys(), key=int)
    xs = np.array([100.0 * int(l) / d['NL'] for l in ls])
    ys = np.array([d['cos'][l] for l in ls])
    ax.plot(xs, ys, color=COL[i], linewidth=2.2, marker='o', markersize=5,
            markeredgecolor='white', markeredgewidth=0.7,
            label=d['nom'].replace('\n', ' '))
    j = ls.index(str(d['L']))
    ax.plot([xs[j]], [ys[j]], marker='o', markersize=15, markerfacecolor='none',
            markeredgecolor=COL[i], markeredgewidth=2.4, zorder=5)
    offs = [(0, 26), (0, 26), (0, -32), (-74, 16), (78, 16)]
    ax.annotate('capa %d: %s' % (d['L'], es(ys[j], 3)),
                xy=(xs[j], ys[j]), xytext=offs[i], textcoords='offset points',
                fontsize=11, color=COL[i], ha='center', fontweight='bold')
ax.text(1.5, 0.142, 'banda de cuasi-ortogonalidad', fontsize=11.5, color=INK2,
        style='italic', va='top')
style(ax)
ax.grid(axis='both', color=GRID, linewidth=0.8, alpha=0.7)
ax.set_xlabel('Profundidad relativa en la red (%)')
ax.set_ylabel(r'$\cos(\mathbf{w}_{ag},\ \hat{\mathbf{r}})$')
ax.set_title('La rotación de la dirección se replica en las tres familias\n'
             'El círculo marca la capa de referencia de cada modelo',
             fontsize=15.5, pad=14)
ax.set_xlim(-2, 102)
ax.set_ylim(-0.09, 0.63)
ax.axhline(0, color='#9a998f', linewidth=1)
ax.legend(loc='upper right', frameon=True, fontsize=11.5, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'cmp_rotacion.png'), dpi=150)
fig.savefig(os.path.join(OUT, 'cmp_rotacion.pdf'))
plt.close(fig)

# ---------- FIGURA C: escala y deteccion ----------
fig, (c1, c2) = plt.subplots(1, 2, figsize=(15, 6.4))
v = [d['kappa'] for d in D]
c1.bar(x, v, color=COL, width=0.62)
style(c1)
c1.axhline(1.0, color='#52514e', linestyle='--', linewidth=1.4)
c1.text(len(D) - 0.45, 1.03, r'$\kappa = 1$  (sin contracción)', ha='right',
        fontsize=11, color=INK2)
c1.set_xticks(x)
c1.set_xticklabels(names, fontsize=11)
c1.set_ylabel(r'$\kappa = s_{ag} / s_{chat}$')
c1.set_title('Contracción de escala:\nel formato agéntico comprime el flujo residual',
             fontsize=14, pad=12)
for i, val in enumerate(v):
    c1.text(i, val + 0.03, es(val, 3), ha='center', fontsize=12.5, color=INK)
c1.set_ylim(0, 1.18)

v2 = [d['aurc'] - d['aura'] for d in D]
c2.bar(x, v2, color=COL, width=0.62)
style(c2)
c2.set_xticks(x)
c2.set_xticklabels(names, fontsize=11)
c2.set_ylabel('Gap de detección  (AUROC chat − AUROC agéntico)')
c2.set_title('Gap de detección:\nvaría por modelo, no es universal', fontsize=14, pad=12)
for i, val in enumerate(v2):
    c2.text(i, val + 0.007, es(val, 3), ha='center', fontsize=12.5, color=INK)
c2.set_ylim(0, 0.25)
fig.suptitle('Dos consecuencias del cambio de formato, por modelo', fontsize=16.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, 'cmp_escala_y_deteccion.png'), dpi=150)
fig.savefig(os.path.join(OUT, 'cmp_escala_y_deteccion.pdf'))
plt.close(fig)

print('figuras escritas en', OUT)
for f in sorted(os.listdir(OUT)):
    print('  ', f, os.path.getsize(os.path.join(OUT, f)), 'B')

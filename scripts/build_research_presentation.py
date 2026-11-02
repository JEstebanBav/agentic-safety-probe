"""Build the English research presentation from the manuscript's saved evidence."""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

from assemble_article import bib_field

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / 'docs/paper'
OUT = PAPER / 'Research_Presentation.pptx'
M = json.loads((PAPER / 'evidence/metrics.json').read_text(encoding='utf-8'))
T = json.loads((PAPER / 'evidence/table_transcription.json').read_text(encoding='utf-8'))
KEYS = ['llama', 'mistral', 'qwen8b', 'qwen32b']
AUTHORS = 'Juan Esteban Espitia Leiva and Helen Stefany Penagos Galvis'
NAVY, TEAL, INK, GRAY, PALE, WHITE = '172E45', '176D78', '233544', '647483', 'F0F5F7', 'FFFFFF'
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333333), Inches(7.5)
prs.core_properties.title = 'Transformation of the Refusal Direction in Agentic Language Models'
prs.core_properties.author = AUTHORS
prs.core_properties.last_modified_by = AUTHORS
prs.core_properties.subject = 'Research AI Safety'
prs.core_properties.keywords = 'AI safety, mechanistic interpretability, refusal, agentic language models'
prs.core_properties.comments = ''


def text(slide, x, y, w, h, value, size=21, color=INK, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(value.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name, p.font.size = 'Aptos', Pt(size)
        p.font.bold, p.font.color.rgb = bold, RGBColor.from_string(color)
        p.space_after, p.line_spacing = Pt(9), 1.1
    return shape


def box(slide, x, y, w, h, fill=PALE):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.fill.background()
    return shape


def new(title, section, source=''):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(WHITE)
    box(slide, 0, 0, 0.13, 7.5, TEAL)
    text(slide, .55, .25, 11.9, .3, section.upper(), 12, TEAL, True)
    text(slide, .55, .76, 12.15, 1.35 if '\n' in title else 1.15, title, 31, NAVY, True)
    box(slide, .55, 7.02, 12.2, .014, 'D7E1E7')
    text(slide, .55, 7.12, 11.75, .2, 'REFUSAL DIRECTION IN AGENTIC LANGUAGE MODELS', 9, GRAY)
    text(slide, 12.1, 7.09, .6, .25, f'{len(prs.slides):02}', 11, GRAY)
    if source:
        slide.notes_slide.notes_text_frame.text = source
    return slide


def points(slide, items, x=.65, y=1.96, w=11.95, h=4.65, size=22):
    return text(slide, x, y, w, h, '\n'.join('• ' + item for item in items), size)


def takeaway(slide, message, y=6.28):
    box(slide, .55, y, 12.2, .58)
    text(slide, .74, y + .12, 11.78, .4, message, 16, TEAL, True)


def table(slide, headers, rows, widths=None, y=2.0, row_h=.66, size=19):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(.65), Inches(y), Inches(12.0), Inches(row_h * (len(rows) + 1)))
    tbl = shape.table
    if widths:
        for c, width in zip(tbl.columns, widths):
            c.width = Inches(width)
    for ri, values in enumerate([headers] + rows):
        for ci, value in enumerate(values):
            cell = tbl.cell(ri, ci)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(.11)
            cell.margin_top = cell.margin_bottom = Inches(.10)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(NAVY if ri == 0 else (PALE if ri % 2 else WHITE))
            for p in cell.text_frame.paragraphs:
                p.font.name = 'Aptos'
                p.font.size = Pt(size if ri else size - 1)
                p.font.bold = ri == 0
                p.font.color.rgb = RGBColor.from_string(WHITE if ri == 0 else INK)
                p.space_after = Pt(0)
    return shape


def cards(slide, left_title, left_items, right_title, right_items, size=21):
    for x, heading, items in [(.55, left_title, left_items), (6.78, right_title, right_items)]:
        box(slide, x, 1.93, 5.97, 4.10)
        text(slide, x + .22, 2.15, 5.52, .55, heading, 23, TEAL, True)
        points(slide, items, x + .22, 2.94, 5.48, 2.9, size)


def figure(title, name, section, caption, source):
    slide = new(title, section, source)
    path = PAPER / 'figures' / name
    iw, ih = Image.open(path).size
    width = min(12.1, 4.30 * iw / ih)
    height = width * ih / iw
    slide.shapes.add_picture(str(path), Inches((13.333333 - width) / 2), Inches(1.80 + (4.30 - height) / 2), width=Inches(width))
    takeaway(slide, caption)
    return slide


def fmt(value, places=3):
    return f'{value:.{places}f}'


def ci(values):
    return '[' + ', '.join(fmt(v) for v in values) + ']'


def main():
    s = new('Transformation of the Refusal Direction\nin Agentic Language Models', 'Research AI Safety')
    text(s, .65, 2.35, 12, 1.15, 'Juan Esteban Espitia Leiva\nHelen Stefany Penagos Galvis', 25, NAVY, True)
    text(s, .65, 3.75, 12, .4, 'Research AI Safety', 23, TEAL)
    text(s, .65, 4.55, 11.8, 1.12, 'A controlled comparison of refusal-related representations\nunder conversational and tool-enabled prompting', 25)
    takeaway(s, '96 harmful–benign pairs  •  4 open-weight models  •  3 levels of harmful-intent explicitness')

    s = new('Does a conversational safety signal transfer to agents?', 'Research question', 'Article, Sections 1.1–1.4 and 6.1.')
    text(s, .70, 2.0, 11.8, 1.65, 'When a language model moves from conversation to an agentic format, does its refusal direction retain the ability to detect harmful intent, including concealed intent?', 29, NAVY)
    points(s, ['The user instruction stays identical across formats.', 'The system message and tool schemas change.', 'Projection, discrimination and observed refusal are evaluated separately.'], y=4.1, h=1.85, size=22)
    takeaway(s, 'The signal transfers partially, with lower discrimination in all four models.')

    s = new('The closest studies address different parts of the problem', 'Related research', 'Arditi et al. (2024); Andriushchenko et al. (2024/2025); Lermen et al. (2024); Pinto (2026, v1); Zhong and Li (2026). Full citations appear on slides 35–36 and in references.bib.')
    table(s, ['Study', 'Main focus'], [
        ['Arditi et al. (2024)', 'A refusal-mediating direction in conversational models'],
        ['AgentHarm (2024/2025)', 'Harmful behaviour in tool-using agents'],
        ['Lermen et al. (2024)', 'Refusal-vector ablation in Llama agents'],
        ['Pinto (2026, v1)', 'Chat-to-agent transfer of additive activation steering'],
        ['Zhong & Li (2026)', 'Refusal in relation to model-persona representations'],
    ], [3.25, 8.75], row_h=.65, size=19)
    takeaway(s, 'Our study links representation transfer, intent explicitness and model-dependent behavioural effects.')

    s = new('Our contribution is a controlled test of format transfer', 'Contribution', 'Article, Sections 1.3, 2.3 and 6.2.')
    points(s, ['Identical instruction text in chat and agentic formats, with matched harmful–benign counterparts.', 'Three explicitness strata: explicit, contextual and legitimately framed harmful intent.', 'Controls for benign shifts, scale contraction, circular estimation and layer selection.', 'Separate tests of geometric change, observed refusal, causal intervention and agent-trained monitors.'], size=23)
    takeaway(s, 'The contribution concerns this controlled corpus; it does not establish universal deployment reliability.')

    s = new('A paired 2 × 2 design isolates the format contrast', 'Study design', 'Article, Sections 3.2.1 and 3.3.1.')
    table(s, ['Instruction', 'Conversational format', 'Agentic format'], [
        ['Harmful', 'Harmful request + chat scaffold', 'Same request + agent scaffold'],
        ['Benign', 'Benign counterpart + chat scaffold', 'Same counterpart + agent scaffold'],
    ], [2.05, 4.95, 5.0], y=2.15, row_h=1.0, size=22)
    points(s, ['Both arms receive the same scaffold within each format.', 'Every pair retains a common identifier throughout extraction and evaluation.'], y=5.0, h=1.1, size=21)
    takeaway(s, '96 pairs produce 384 primary stimuli; tools are described but never executed.')

    s = new('The corpus combines new pairs and adapted benchmark tasks', 'Corpus construction', 'Article, Section 3.2; Table 1. Corpus: data/dataset_v2.jsonl; composition report: data/dataset_v2_report.json.')
    cards(s, 'Composition', ['96 harmful–benign pairs', '62 authored for this study', '34 adapted from AgentHarm', '7 harm categories'], 'Construction criteria', ['Match topic across both arms', 'Keep user text identical across formats', 'Record origin and explicitness', 'Revise or discard unsuitable pairs'], size=21)
    takeaway(s, 'Adapted counterparts serve the paired design; this is not a full evaluation of the published benchmark.')

    s = new('Explicitness describes the harmful member of each pair', 'Corpus construction', 'Article, Sections 3.2.2–3.2.3; Tables 1–2. Model-assisted review used Claude; final decisions remained with the corpus researcher.')
    table(s, ['Stratum', 'Pairs', 'What the instruction reveals'], [
        ['Explicit', 42, 'Harmful intent is openly expressed'],
        ['Contextual', 24, 'Harmfulness must be inferred from context'],
        ['Legitimately framed', 30, 'Harmful intent appears within a legitimate pretext'],
    ], [3.0, 1.05, 7.95], row_h=.85, size=21)
    points(s, ['One researcher consolidated labels with model-assisted review.', 'Independent second annotation and agreement measurement remain pending.'], y=5.15, h=1.0, size=20)
    takeaway(s, 'A stated pretext or claim of permission does not verify actual authorisation.')

    s = new('Scaffold and lexical checks support the paired comparison', 'Corpus controls', 'Article, Sections 3.2.1–3.2.3 and 3.5; Tables 1–2.')
    points(s, ['A catalogue of 64 simulated tools; each task uses 2–8 tools, averaging 3.4.', 'Within each format, harmful and benign counterparts share the system message and catalogue.', 'Median token-set Jaccard similarity by stratum: 0.408, 0.413 and 0.422.', 'Names and accounts are fictitious; generated tool calls have no external effect.'], size=24)
    takeaway(s, 'These checks reduce topic and scaffold confounds; independent label validation is still needed.')

    s = new('Four models share the same 96-pair protocol', 'Models and configuration', 'Article, Tables 3–4; saved run configurations. Layer indices are zero-based.')
    table(s, ['Model', 'Total layers', 'Layers read', 'Reference layer', 'Precision'], [
        [M[k]['modelo'], M[k]['capas_total'], len(M[k]['capas_evaluadas']), M[k]['capa_referencia'], M[k]['precision']] for k in KEYS
    ], [3.6, 2.1, 2.1, 2.7, 1.5], size=20)
    points(s, ['Fixed model weights and seed 42; separate stimulus passes.', 'Generation completed for Llama, Mistral and Qwen3-8B; Qwen3-32B has representation results only.'], y=5.45, h=1.35, size=20)

    s = new('We read the residual stream at the final prompt token', 'Activation measurement', 'Article, Section 3.3.1 and Equations (1)–(4); Section 3.3.3.')
    cards(s, 'Where we measure', ['Residual-stream activations at swept layers', 'The final token of the rendered input', 'Before the first response token is generated'], 'Why this position', ['Can attend to the full preceding input', 'Includes system, tools and delimiters', 'Has the same predictive role in both formats'], size=21)
    takeaway(s, 'The final rendered token may be a template delimiter; it need not be the last user word.')

    s = new('The refusal direction is a normalised mean contrast', 'Representation estimator', 'Article, Section 3.3.1, Equations (5) and (7). Following Arditi et al. (2024).')
    box(s, .65, 2.05, 12, 1.18)
    text(s, .95, 2.35, 11.35, .5, 'r = mean(x_harmful, chat) − mean(x_benign, chat)     r̂ = r / ||r||', 26, NAVY, True)
    text(s, .90, 3.65, 11.5, .65, 'Projection at layer l:     p = x · r̂', 29, TEAL, True)
    points(s, ['The pair contrast reduces topic-related variation.', 'Directions are estimated without the pairs they later score.', 'A readable harmfulness contrast is distinct from an observed refusal decision.'], y=4.65, h=1.55, size=22)
    takeaway(s, 'The geometric estimator is a diagnostic; its causal relevance requires a separate intervention test.')

    s = new('The benign arm separates a general shift from a harm effect', 'Format contrast', 'Article, Section 3.3.1, Equations (8)–(10); Table 6. Positive delta denotes lower projection in agentic format.')
    text(s, .85, 2.15, 11.7, .6, 'Δp_h = p_h,chat − p_h,agent', 31, NAVY, True)
    text(s, .85, 3.05, 11.7, .6, 'Δp_b = p_b,chat − p_b,agent', 31, NAVY, True)
    text(s, .85, 4.10, 11.7, .6, 'Content × format interaction:   Δ²p = Δp_h − Δp_b', 29, TEAL, True)
    points(s, ['A rigid translation of both classes can lower raw projection without weakening harmful–benign discrimination.'], y=5.35, h=.8, size=22)
    takeaway(s, 'We test within-pair changes and report their uncertainty rather than interpreting raw shifts alone.')

    s = new('Scale contraction and discrimination are different quantities', 'Measurement controls', 'Article, Section 3.3.1, Equations (12)–(15); Table 7.')
    cards(s, 'Scale-dependent measurements', ['Raw projection and centroid separation', 'Contraction: κ = s_agent / s_chat', 'Residual norms differ across models'], 'Scale-invariant checks', ['Standardised separation: d = S / s', 'AUROC depends on score ordering', 'Positive affine rescaling leaves both unchanged'], size=21)
    takeaway(s, 'Only loss that survives in standardised separation and AUROC supports reduced discriminative margin.')

    s = new('Pair-held-out evaluation corrects circular estimation', 'Validation', 'Article, Sections 3.4.1–3.4.2; Table 13. Cross-fitting uses five folds by pair. Logistic probes use repeated pair-wise cross-validation.')
    points(s, ['Five folds keep all variants of a pair together.', 'Estimate the direction on training pairs and score only held-out pairs.', 'Report bootstrap intervals and paired tests; use shuffled labels as a null reference.', 'Separate in-sample diagnostics from held-out evidence and correct repeated-CV uncertainty.'], size=24)
    takeaway(s, 'A perfect training AUROC in high dimension is insufficient evidence of generalisation.')

    s = new('Layer selection and inference use separate pairs', 'Validation', 'Article, Section 4.2.2; Table 13. Reference-layer selection is repeated across 200 random pair partitions.')
    table(s, ['Model', 'Selection pairs', 'Inference pairs', 'Chosen layer'], [
        [M[k]['modelo'], M[k]['seleccion']['n_seleccion'], M[k]['seleccion']['n_inferencia'], M[k]['seleccion']['capa']] for k in KEYS
    ], [4.2, 2.6, 2.6, 2.6], size=20)
    points(s, ['Inference-half interactions exclude zero in every model.', 'All 200 partitions select the same reference layer within each model.'], y=5.6, h=1.05, size=20)
    s.notes_slide.notes_text_frame.text += '\nRepeated partitions assess stability on this corpus, not independent replication. Bonferroni thresholds reflect the number of layers evaluated in each model.'

    s = new('Held-out discrimination falls in every model', 'Main results', 'Article, Table 5. AUROC gap = chat minus agent; 95% paired bootstrap intervals. All models use the same 96 pairs.')
    table(s, ['Model', 'Chat AUROC', 'Agent AUROC', 'Gap [95% CI]'], [
        [M[k]['modelo'], fmt(M[k]['deteccion']['auroc_chat']), fmt(M[k]['deteccion']['auroc_agente']), fmt(M[k]['deteccion']['brecha']) + ' ' + ci(M[k]['deteccion']['brecha_ic'])] for k in KEYS
    ], [3.25, 2.35, 2.35, 4.05], size=20)
    takeaway(s, 'The conversational direction retains signal, with a smaller harmful–benign margin agentically.')

    figure('The format effect changes across model depth', 'fig_layers.png', 'Main results', 'Layer profiles use repeated measurements of the same corpus; layers are not independent samples.', 'Article, Figure 1; held-out AUROC and paired projection profiles. Both formats are scored using the conversational direction.')

    s = new('The harmful shift exceeds the benign shift', 'Benign-control results', 'Article, Table 6. Raw projection units are model-specific; magnitudes should not be ranked across models.')
    table(s, ['Model', 'Δp harmful', 'Δp benign', 'Interaction [95% CI]'], [
        [M[k]['modelo'], fmt(M[k]['desplazamiento']['dp_h'], 2), fmt(M[k]['desplazamiento']['dp_b'], 2), fmt(M[k]['desplazamiento']['interaccion'], 2) + ' ' + ci(M[k]['desplazamiento']['interaccion_ic'])] for k in KEYS
    ], [3.25, 2.1, 2.1, 4.55], size=19)
    takeaway(s, 'Positive content × format interactions support a harmfulness-specific component in all four models.')

    s = new('The loss persists after accounting for contraction', 'Scale-control results', 'Article, Table 7. Raw separation loss and standardised separation loss are different quantities.')
    table(s, ['Model', 'κ', 'd chat', 'd agent', 'Loss in d'], [
        [M[k]['modelo'], fmt(M[k]['escala']['kappa']), fmt(M[k]['escala']['d_chat']), fmt(M[k]['escala']['d_agente']), f"{100*M[k]['escala']['perdida_estandarizada']:.1f}%"] for k in KEYS
    ], [3.6, 1.8, 2.2, 2.2, 2.2], size=21)
    takeaway(s, 'Pure rescaling cannot explain the simultaneous decline in standardised separation and AUROC.')

    figure('Contextual harmful intent is the hardest to discriminate', 'fig_strata.png', 'Explicitness results', 'Bars are point estimates; differences between strata are not formal tests of a stratum interaction.', 'Article, Figure 2 and Table 8. Explicit requests retain the highest agentic AUROC in every model. Contextual requests have the largest AUROC loss, tied with framed requests in Mistral. The explicit stratum has the largest proportional loss in d; the ordering depends on the metric.')

    figure('Chat and agentic contrasts are poorly aligned', 'fig_rotation.png', 'Reorientation', 'Directions estimated by means and by logistic regression are separate estimators with different reliability.', 'Article, Figure 4 and Table 10. Mean-difference directions have within-format reliability approximately 0.92–0.97 but low cross-format alignment. Reliability-normalised angles are diagnostic estimates, not exact population angles.')

    figure('The conversational contrast becomes a dominant PCA axis', 'fig_pca.png', 'Geometric validation', 'PCA describes sample geometry; alignment with PC1 does not establish a causal refusal mechanism.', 'Article, Figure 5 and Table 12. Harmful and benign activations are concatenated and centred by the global mean. Classes are not centred separately, which would remove the centroid contrast. PC1 alignment exceeds 0.95 after model-specific depths.')

    figure('Tool schemas reproduce most of the full-format effect', 'fig_factorial.png', 'Factorial decomposition', 'The schemas also add most tokens; content and length effects remain confounded.', 'Article, Figure 3 and Table 9. Tool-only scaffolds reproduce 97–99% of the full interaction in Llama and both Qwens, and 68% in Mistral. Effects are non-additive. The factorial uses full-sample directions and is descriptive; the existence of the main effect is established out of sample.')

    figure('Observed refusal decreases under agentic prompting', 'fig_behavior.png', 'Behavioural results', 'First-turn heuristic labels and simulated tool calls provide exploratory behavioural evidence.', 'Article, Figure 6 and Table 14. Error bars are 95% Wilson intervals. Tools are never executed. Qwen3-8B has incomplete reasoning outputs; non-refusal is not equivalent to completed compliance.')

    s = new('The refusal drop appears in all three completed models', 'Behavioural results', 'Article, Section 4.3 and Table 14. Qwen3-32B has no completed behavioural evaluation. Denominators below are all 96 requests, including Qwen3-8B incomplete responses.')
    table(s, ['Model', 'Harmful: chat → agent', 'Benign: chat → agent'], [
        [M[k]['modelo'], f"{M[k]['conducta']['rechazo']['chat_harmful'][0]}/96 → {M[k]['conducta']['rechazo']['agent_harmful'][0]}/96", f"{M[k]['conducta']['rechazo']['chat_benign'][0]}/96 → {M[k]['conducta']['rechazo']['agent_benign'][0]}/96"] for k in KEYS[:3]
    ], [3.6, 4.2, 4.2], row_h=.85, size=23)
    points(s, ['Llama: 40 pairs stop refusing; one changes in the opposite direction.', 'Mistral: 42 versus four; Qwen3-8B: 31 versus zero among 63 complete pairs.'], y=5.6, h=1.05, size=20)

    s = new('Incomplete Qwen responses limit behavioural interpretation', 'Coverage and uncertainty', 'Article, Sections 4.3–4.4 and 5.5. Qwen3-8B has 32 harmful-agent outputs with incomplete reasoning. Its paired test uses 63 cases complete in both formats.')
    cards(s, 'Qwen3-8B', ['32 harmful-agent outputs have incomplete reasoning', 'Tool calls and refusal labels do not measure executed harm', 'Intervention is not distinguishable from random control'], 'Qwen3-32B', ['Representation analyses are complete', 'Behavioural generation did not complete', 'No behavioural or causal result is claimed'], size=21)
    takeaway(s, 'Geometric evidence has broader model coverage than the behavioural and causal evidence.')

    s = new('Intervention restores the measured lost projection', 'Causal test', 'Article, Section 4.4 and Table 15. Llama layers 76, 56 and 28 were declared before execution; Mistral and Qwen3-8B use separately selected layers.')
    text(s, .85, 2.0, 11.7, .7, 'Residual intervention:     x′ = x + αr̂', 32, NAVY, True)
    points(s, ['Dose one adds the layer-specific mean projection lost under format change.', 'Test harmful requests initially complied with agentically.', 'Compare against a random direction with matched norm.', 'Apply the same refusal-direction addition to benign requests to assess selectivity.'], y=3.1, h=2.85, size=23)
    takeaway(s, 'The measured projection shift matches α; behavioural change is evaluated separately.')

    figure('Causal effects depend on model, layer and dose', 'fig_intervention.png', 'Causal results', 'Larger doses can induce benign refusal or degeneration; more intervention is not necessarily better.', 'Article, Figure 7 and Table 15. Refusal rates use initially complying subsets, not all 96 pairs. Error bars are 95% Wilson intervals. Qwen3-8B responses can remain incomplete.')

    s = new('Restoring projection has a selective effect only in some settings', 'Causal results', 'Article, Section 4.4 and Table 15. Counts are restored refusals among initially complying harmful requests; controls receive a matched-norm random direction. Benign counts measure induced refusal.')
    causal_specs = [('llama', '76', '1'), ('llama', '56', '1'), ('mistral', '30', '2'), ('qwen8b', '32', '1')]
    rows = []
    for key, layer, dose in causal_specs:
        v = M[key]['causal'][layer]['dosis'][dose]
        ratio = lambda name: f"{v[name][0]}/{v[name][1]}"
        rows.append([M[key]['modelo'], layer, dose + '×', ratio('restaura'), ratio('control'), ratio('benigno')])
    table(s, ['Model', 'Layer', 'Dose', 'Harmful', 'Random', 'Benign'], rows, [3.5, 1.3, 1.3, 2.0, 2.0, 1.9], size=21)
    takeaway(s, 'This supports a partial causal role in Llama, a weaker effect in Mistral, and no detected Qwen3-8B effect.')

    s = new('Agent-trained monitors recover a readable signal', 'Monitoring results', 'Article, Table 11. Five pair-wise folds; thresholds calibrated using each fold’s training data. AUROC is averaged across folds. DM: difference of means; LR: logistic regression.')
    table(s, ['Model', 'DM AUROC', 'LR AUROC', 'LR F1', 'LR FPR'], [
        [M[k]['modelo'], fmt(M[k]['monitores']['medias']['auroc']), fmt(M[k]['monitores']['logistica']['auroc']), fmt(M[k]['monitores']['logistica']['f1']), fmt(M[k]['monitores']['logistica']['fpr'])] for k in KEYS
    ], [3.6, 2.1, 2.1, 2.1, 2.1], size=21)
    takeaway(s, 'Within-corpus recovery does not show that the model uses this axis to refuse or that it transfers externally.')

    s = new('Geometric readability does not fully explain refusal decisions', 'Interpretation', 'Article, Section 4.3 and Table 16. Llama projection predicts refusal with AUROC 0.972 in chat and 0.971 agentically, but drop magnitude does not separate requests changing behaviour from those retaining refusal (Mann–Whitney p = 0.48).')
    cards(s, 'What the evidence supports', ['Harmfulness remains readable agentically', 'The chat direction loses discriminative margin', 'Some interventions causally alter refusal'], 'What remains unresolved', ['Why individual requests change behaviour', 'How the model internally reads the signal', 'Whether monitors generalise beyond this corpus'], size=21)
    takeaway(s, 'Projection, discrimination, refusal and intervention effects answer different research questions.')

    s = new('The refusal representation changes rather than disappearing', 'Conclusions', 'Article, Sections 6.1–6.2.')
    points(s, ['The conversational direction retains partial discriminative power under agentic prompting.', 'The loss survives benign controls, scale-invariant checks and out-of-sample validation.', 'The contrast reorients across formats; contextual intent is least discriminable agentically.', 'Agent-trained monitors recover signal within this corpus, while causal refusal effects remain model-dependent.'], size=23)
    takeaway(s, 'Conversational calibration is insufficient evidence that a safety monitor will transfer to an agentic setting.')

    s = new('The conclusions are bounded by the experimental design', 'Limitations', 'Article, Sections 5.5 and 6.3.')
    points(s, ['A small, English-only corpus with one independent annotator.', 'Schema content, extra length and template changes are not fully separated.', 'Short first-turn generations, heuristic labels and incomplete Qwen reasoning.', 'No external corpus, real tool execution, multi-step agent evaluation or repeated-run uncertainty estimate.', 'Four models do not establish a scaling law or universal mechanism.'], size=23)

    s = new('Next tests should target the remaining validity questions', 'Research priorities and reproducibility', 'Article, Sections 3.3.3, 3.5, 6.3 and 7. Saved configurations, per-pair activations and generations allow statistical reanalysis without repeating inference. Low-precision execution can vary across hardware.')
    points(s, ['Independently audit harmfulness and explicitness labels, including ambiguous permission claims.', 'Compare length-matched schemas, irrelevant schemas and controlled prompt templates.', 'Evaluate agent-calibrated monitors on an independent corpus with false-positive analysis.', 'Extend behavioural evaluation to complete reasoning, additional models and multi-step contexts.', 'Retain pair identifiers, configurations and saved outputs; keep tools simulated during controlled tests.'], size=22)

    bib = (PAPER / 'references.bib').read_text(encoding='utf-8')
    entries = {}
    for match in re.finditer(r'@\w+\{([^,]+),', bib):
        end = bib.find('\n@', match.end())
        entries[match.group(1)] = bib[match.start():end if end != -1 else len(bib)]
    ref_keys = ['arditi2024', 'andriushchenko2024', 'lermen2024', 'pinto2026', 'zhong2026', 'zou2023', 'park2024', 'alain2017', 'templeton2024', 'greenblatt2024', 'ouyang2022', 'bai2022', 'olah2020', 'nadeau2003']
    assert all(key in entries for key in ref_keys)
    for index, batch in enumerate([ref_keys[:7], ref_keys[7:]], 1):
        s = new(f'References ({index}/2)', 'Research literature')
        sources = []
        for row, key in enumerate(batch):
            entry = entries[key]
            author = bib_field(entry, 'author').split(' and ')[0].split(',')[0]
            value = f"{author} et al. ({bib_field(entry, 'year')}). {bib_field(entry, 'title')}."
            if key == 'pinto2026':
                value += ' Version 1.'
            text(s, .7, 1.84 + row * .70, 11.9, .65, value, 18)
            url = bib_field(entry, 'url') or 'https://doi.org/' + bib_field(entry, 'doi')
            sources.append(value + '\n' + url)
        s.notes_slide.notes_text_frame.text = '\n\n'.join(sources)

    assert len(prs.slides) == 36, len(prs.slides)
    prs.save(OUT)
    print(json.dumps({'powerpoint': str(OUT), 'slides': len(prs.slides), 'bytes': OUT.stat().st_size}))


if __name__ == '__main__':
    main()

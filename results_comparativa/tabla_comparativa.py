import json, io, os
P = r"C:/Users/38614180/AgentHarm/agentic-safety-probe"
MOD = [
 ('results_llama33',        'Llama-3.3-70B-Instruct', 'FP16',  80),
 ('results_qwen32b',        'Qwen3-32B',              '4-bit', 64),
 ('results_qwen8b',         'Qwen3-8B',               'FP16',  36),
 ('results_mistral7b_4bit', 'Mistral-7B-v0.3',        '4-bit', 32),
 ('results_mistral7b_fp16_v2','Mistral-7B-v0.3',      'FP16',  32),
]
out = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'master.txt'),'w',encoding='utf-8')
def w(s=''): out.write(str(s)+'\n')

rows = []
for d, nom, prec, NL in MOD:
    o = json.load(io.open(P+'/'+d+'/results.json', encoding='utf-8'))
    rob = o.get('robust') or {}
    L = str(rob.get('layer_selection',{}).get('selected_layer'))
    fe = (o.get('format_effect') or {}).get('per_layer', {}).get(L, {})
    rb = rob.get('per_layer', {}).get(L, {})
    ad = (o.get('agent_direction') or {}).get('layer_results', {}).get(L, {})
    la = o.get('linear_aurocs', {}).get(L)
    lastd = o.get('linear_aurocs_std', {}).get(L)
    rows.append(dict(dir=d, nom=nom, prec=prec, NL=NL, L=int(L), depth=100*int(L)/NL,
        sonda=la, sonda_sd=lastd,
        dph=fe.get('delta_p_harmful'), dpb=fe.get('delta_p_benign'), inter=fe.get('interaction'),
        inter_d=fe.get('interaction_cohens_d'), inter_p=fe.get('interaction_p'), phi=fe.get('common_shift_fraction'),
        dchat=fe.get('cohens_d_chat'), dag=fe.get('cohens_d_agent'),
        loss_raw=fe.get('separation_loss_raw'), loss_std=fe.get('separation_loss_standardised'),
        kappa=fe.get('scale_contraction'),
        aur_c=fe.get('auroc_chat'), aur_a=fe.get('auroc_agent'),
        dpn=rb.get('delta_p_naive'), dpx=rb.get('delta_p_crossfit'), bias=rb.get('bias_estimate'),
        ci=(rb.get('ci_lower_crossfit'), rb.get('ci_upper_crossfit')), dz=rb.get('cohens_d_paired_crossfit'),
        ndec=rb.get('n_decreasing'), npairs=rb.get('n_pairs'), signp=rb.get('sign_test_p'),
        reliab=rb.get('split_half_reliability'), auroc_wag_oof=rb.get('auroc_w_agent_oof'),
        f1=rb.get('monitor_f1_oof'), fpr=rb.get('monitor_fpr_oof'),
        prec_oof=rb.get('monitor_precision_oof'), rec_oof=rb.get('monitor_recall_oof'),
        cos=ad.get('cosine_similarity') if isinstance(ad,dict) else None))

w('TABLA MAESTRA - capa seleccionada por division de muestra (Bonferroni)')
w('='*150)
w('%-22s %-6s %-5s %-7s %-8s %-9s %-9s %-9s %-7s %-7s' % ('modelo','prec','capa','prof.%','sonda VC','dp_h','dp_b','Delta2p','d_int','phi%'))
w('-'*150)
for r in rows:
    w('%-22s %-6s %-5d %-7.0f %-8.3f %-9.3f %-9.3f %-9.3f %-7.2f %-7.1f' % (
        r['nom'], r['prec'], r['L'], r['depth'], r['sonda'] or 0, r['dph'] or 0, r['dpb'] or 0,
        r['inter'] or 0, r['inter_d'] or 0, 100*(r['phi'] or 0)))
w()
w('%-22s %-6s %-8s %-8s %-9s %-9s %-8s %-9s %-9s' % ('modelo','prec','d_chat','d_agent','perd_std%','perd_cru%','kappa','AUROCch','AUROCag'))
w('-'*150)
for r in rows:
    w('%-22s %-6s %-8.3f %-8.3f %-9.1f %-9.1f %-8.3f %-9.3f %-9.3f' % (
        r['nom'], r['prec'], r['dchat'] or 0, r['dag'] or 0, 100*(r['loss_std'] or 0), 100*(r['loss_raw'] or 0),
        r['kappa'] or 0, r['aur_c'] or 0, r['aur_a'] or 0))
w()
w('%-22s %-6s %-9s %-9s %-8s %-20s %-8s %-8s' % ('modelo','prec','dp_naive','dp_cross','sesgo%','IC95 crossfit','d_z','pares dec.'))
w('-'*150)
for r in rows:
    b = 100*abs(r['bias'] or 0)/max(abs(r['dpx'] or 1),1e-9)
    w('%-22s %-6s %-9.3f %-9.3f %-8.1f [%7.3f, %7.3f]   %-8.2f %d/%d' % (
        r['nom'], r['prec'], r['dpn'] or 0, r['dpx'] or 0, b, r['ci'][0] or 0, r['ci'][1] or 0,
        r['dz'] or 0, r['ndec'] or 0, r['npairs'] or 0))
w()
w('%-22s %-6s %-9s %-9s %-10s %-8s %-8s %-8s' % ('modelo','prec','cos(w,r)','reliab','AUROCwag','F1 oof','FPR oof','gap det.'))
w('-'*150)
for r in rows:
    w('%-22s %-6s %-9.4f %-9.3f %-10.3f %-8.3f %-8.3f %-8.3f' % (
        r['nom'], r['prec'], r['cos'] or 0, r['reliab'] or 0, r['auroc_wag_oof'] or 0,
        r['f1'] or 0, r['fpr'] or 0, (r['aur_c'] or 0)-(r['aur_a'] or 0)))
out.close()
print(io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'master.txt'),encoding='utf-8').read())

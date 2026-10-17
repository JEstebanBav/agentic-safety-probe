"""Guion de la versión ligera, con miniaturas y las mismas notas del PowerPoint."""
import html
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

COMUN = Path(__file__).resolve().parents[2] / "sustentacion/fuente/guion_pdf.py"
spec = importlib.util.spec_from_file_location("guion_comun", COMUN)
comun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comun)


def main(registro, carpeta, salida):
    reg = json.loads(Path(registro).read_text(encoding="utf-8"))
    total = max(r["fin"] for r in reg)
    filas = "".join(f"<tr><td>{r['numero']}</td><td>{html.escape(r['titulo'])}</td><td>{comun.mmss(r['t'])}</td></tr>"
                    for r in reg if not r.get("respaldo"))
    bloques = []
    for r in reg:
        tiempo = "Apoyo para preguntas" if r.get("respaldo") else f"{comun.mmss(r['t'])} · {comun.mmss(r['inicio'])}–{comun.mmss(r['fin'])}"
        bloques.append(f"""<section><div class="cab"><span>{r['numero']:02d}</span><h2>{html.escape(r['titulo'])}</h2><small>{tiempo}</small></div>
<img src="{comun.miniatura(Path(carpeta) / f's{r["numero"]:02d}.png')}">
<h3>Qué contar</h3>{comun.parrafos(r['guion'])}
<h3>Para enlazar con la siguiente</h3>{comun.parrafos(r.get('transicion') or 'Preguntas.')}
<div class="fu"><h3>Fuentes y alcance</h3>{comun.parrafos(r.get('fuentes') or '')}</div></section>""")
    doc = f"""<!doctype html><html lang="es"><meta charset="utf-8"><title>Guion · presentación ligera</title>
<style>
@page {{size:A4; margin:16mm 16mm 18mm}}
body {{font-family:Calibri,Arial,sans-serif; color:#26282a; font-size:11pt; line-height:1.4}}
h1 {{font-size:25pt;line-height:1.1;margin:0 0 8pt}} h2 {{font-size:14pt;margin:0;flex:1}}
h3 {{font-size:9pt;margin:8pt 0 3pt;color:#416879}} p {{margin:0 0 5pt}}
.meta,small {{color:#70777d}} .meta {{margin:9pt 0 16pt}}
table {{width:100%;border-collapse:collapse;font-size:10pt}} td {{padding:4pt;border-bottom:.5pt solid #d9dde0}}
section {{page-break-inside:avoid;break-inside:avoid;margin-top:18pt;padding-top:10pt;border-top:.8pt solid #d9dde0}}
.cab {{display:flex;align-items:baseline;gap:8pt}} .cab span {{color:#9b743e;font-size:16pt}}
img {{display:block;width:70%;margin:9pt 0}} .fu {{color:#70777d;font-size:8.5pt;line-height:1.3}}
</style><h1>Guion de la presentación ligera</h1>
<p>Transformación de la dirección de rechazo en modelos de lenguaje agénticos.</p>
<p class="meta">Juan Esteban Espitia Leiva · Universidad Externado de Colombia.<br>
14 diapositivas principales y 2 de apoyo · duración orientativa: {comun.mmss(total)}.<br>
Las notas coinciden con el PowerPoint. Hay margen para explicar las figuras con pausas.</p>
<table>{filas}</table>{''.join(bloques)}</html>"""
    destino = Path(registro).with_name("guion_ligero.html")
    destino.write_text(doc, encoding="utf-8")
    subprocess.run([comun.EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={Path(salida).resolve()}", destino.resolve().as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    print("Guion ligero PDF:", Path(salida).resolve())


if __name__ == "__main__":
    main(*sys.argv[1:4])

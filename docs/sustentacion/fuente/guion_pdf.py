"""Genera el guion imprimible: miniatura de cada diapositiva con su texto oral completo.

    python guion_pdf.py build/guion.json build/png ../Sustentacion_Espitia_Leiva_guion.pdf

Escribe un HTML en build/ y lo imprime a PDF con Microsoft Edge sin interfaz.
"""
import base64
import html
import io
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def mmss(s):
    return f"{int(s // 60)}:{int(round(s % 60)):02d}"


def miniatura(ruta, ancho=760):
    im = Image.open(ruta).convert("RGB")
    im = im.resize((ancho, round(im.height * ancho / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parrafos(texto):
    return "".join(f"<p>{html.escape(p.strip())}</p>" for p in texto.strip().split("\n") if p.strip())


def main(registro, carpeta_png, salida):
    reg = json.loads(Path(registro).read_text(encoding="utf-8"))
    total = max(r["fin"] for r in reg)
    filas = "".join(
        f"<tr><td class='n'>{r['numero']}</td><td>{html.escape(r['titulo'])}</td><td class='t'>{mmss(r['t'])}</td><td class='t'>{mmss(r['fin'])}</td></tr>"
        for r in reg if not r.get("respaldo"))
    bloques = []
    for r in reg:
        png = Path(carpeta_png) / f"s{r['numero']:02d}.png"
        tiempo = "Respaldo · solo si el jurado pregunta" if r.get("respaldo") else f"{mmss(r['t'])} · de {mmss(r['inicio'])} a {mmss(r['fin'])}"
        extra = ""
        if r.get("transicion"):
            extra += f"<h4>Transición</h4>{parrafos(r['transicion'])}"
        if r.get("fuentes"):
            extra += f"<h4>Fuentes y trazabilidad</h4><div class='fu'>{parrafos(r['fuentes'])}</div>"
        bloques.append(f"""
<section>
  <div class="cab"><span class="num">{r['numero']}</span><h2>{html.escape(r['titulo'])}</h2><span class="tiempo">{tiempo}</span></div>
  <img src="{miniatura(png)}" alt="Diapositiva {r['numero']}">
  <h4>Guion</h4><div class="g">{parrafos(r['guion'])}</div>{extra}
</section>""")
    doc = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Guion de sustentación</title>
<style>
@page {{ size: A4; margin: 16mm 16mm 18mm 16mm; }}
body {{ font-family: 'Franklin Gothic Book', 'Segoe UI', sans-serif; color: #161A20; font-size: 10.5pt; line-height: 1.45; }}
h1 {{ font-family: Georgia, serif; font-weight: normal; font-size: 22pt; margin: 0 0 4pt; }}
.sub {{ font-family: Georgia, serif; font-style: italic; color: #434B57; font-size: 12pt; margin: 0 0 10pt; }}
.meta {{ color: #6E7682; font-size: 9.5pt; margin-bottom: 14pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9.5pt; margin-bottom: 8pt; }}
td {{ border-bottom: 0.5pt solid #E3E6EA; padding: 3pt 4pt; }}
td.n, td.t {{ text-align: right; white-space: nowrap; color: #434B57; }}
section {{ break-inside: avoid; page-break-inside: avoid; margin: 0 0 16pt; padding-top: 10pt; border-top: 0.75pt solid #D3D7DD; }}
.cab {{ display: flex; align-items: baseline; gap: 8pt; margin-bottom: 6pt; }}
.num {{ font-family: 'Franklin Gothic Medium', sans-serif; color: #C98316; font-size: 14pt; min-width: 18pt; }}
h2 {{ font-family: Georgia, serif; font-weight: normal; font-size: 13pt; margin: 0; flex: 1; }}
.tiempo {{ color: #6E7682; font-size: 9pt; white-space: nowrap; }}
img {{ width: 62%; border: 0.5pt solid #D3D7DD; display: block; margin: 2pt 0 6pt; }}
h4 {{ font-family: 'Franklin Gothic Demi', sans-serif; font-weight: normal; font-size: 8.5pt; letter-spacing: 0.8pt; text-transform: uppercase; color: #434B57; margin: 6pt 0 2pt; }}
p {{ margin: 0 0 4pt; }}
.fu {{ color: #6E7682; font-size: 8.5pt; line-height: 1.35; }}
</style></head><body>
<h1>Guion de sustentación</h1>
<p class="sub">Transformación de la dirección de rechazo en modelos de lenguaje agénticos. Un análisis desde la interpretabilidad mecánica.</p>
<p class="meta">Juan Esteban Espitia Leiva · Tutor: Julián Fernando Sánchez · Pregrado en Ciencia de Datos, Universidad Externado de Colombia.<br>
Exposición prevista: {mmss(total)} de 20:00. El texto coincide con las notas del presentador del archivo .pptx; los tiempos son orientativos.</p>
<table>{filas}</table>
{''.join(bloques)}
</body></html>"""
    destino_html = Path(registro).with_name("guion.html")
    destino_html.write_text(doc, encoding="utf-8")
    salida = Path(salida).resolve()
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={salida}", destino_html.resolve().as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    print("Guion PDF:", salida)


if __name__ == "__main__":
    main(*sys.argv[1:4])

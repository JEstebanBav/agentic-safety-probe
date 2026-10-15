"""Abre el .pptx en PowerPoint (COM), lo guarda normalizado y exporta PDF, notas e imágenes.

    python exportar.py entrada.pptx salida.pptx [--pdf a.pdf] [--notas b.pdf] [--png carpeta]

Guardar desde PowerPoint regenera las imágenes de respaldo de las ecuaciones y
deja el archivo escrito por la propia aplicación en que se presentará.
"""
import argparse
import os
from pathlib import Path

import comtypes.client

PP_SAVE_PPTX = 24
PP_PDF, PP_INTENT_PRINT, MSO_FALSE, PP_HANDOUT_VERTICAL = 2, 2, 0, 1
PP_OUT_SLIDES, PP_OUT_NOTES = 1, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("salida")
    ap.add_argument("--pdf")
    ap.add_argument("--notas")
    ap.add_argument("--png")
    ap.add_argument("--ancho", type=int, default=1600)
    ap.add_argument("--solo", help="exportar solo estas diapositivas a PNG, p. ej. 3,7,12")
    a = ap.parse_args()
    pp = comtypes.client.CreateObject("PowerPoint.Application")
    try:
        pres = pp.Presentations.Open(os.path.abspath(a.entrada), True, False, False)
        pres.SaveAs(os.path.abspath(a.salida), PP_SAVE_PPTX)
        if a.pdf:
            pres.ExportAsFixedFormat(os.path.abspath(a.pdf), PP_PDF, PP_INTENT_PRINT, MSO_FALSE,
                                     PP_HANDOUT_VERTICAL, PP_OUT_SLIDES)
        if a.notas:
            pres.ExportAsFixedFormat(os.path.abspath(a.notas), PP_PDF, PP_INTENT_PRINT, MSO_FALSE,
                                     PP_HANDOUT_VERTICAL, PP_OUT_NOTES)
        if a.png:
            carpeta = Path(a.png)
            carpeta.mkdir(parents=True, exist_ok=True)
            solo = {int(x) for x in a.solo.split(",")} if a.solo else None
            if not solo:
                for f in carpeta.glob("s*.png"):
                    f.unlink()
            alto = round(a.ancho * 9 / 16)
            for i in range(1, pres.Slides.Count + 1):
                if solo and i not in solo:
                    continue
                pres.Slides(i).Export(str((carpeta / f"s{i:02d}.png").resolve()), "PNG", a.ancho, alto)
        print("Diapositivas:", pres.Slides.Count)
        pres.Close()
    finally:
        pp.Quit()


if __name__ == "__main__":
    main()

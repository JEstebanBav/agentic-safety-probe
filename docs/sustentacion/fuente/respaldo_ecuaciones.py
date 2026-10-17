"""Da a cada ecuación OMML una imagen de respaldo, como hace PowerPoint con las suyas.

    python respaldo_ecuaciones.py build/con_ecuaciones.pptx build/con_respaldo.pptx

PowerPoint muestra la ecuación editable (mc:Choice). Los visores que no admiten
ecuaciones (Google Slides, Keynote, versiones antiguas de LibreOffice) usan el
contenido alternativo (mc:Fallback); sin este paso verían el marcador de texto.
PowerPoint exporta cada ecuación a PNG y el PNG se inserta en el respaldo.
"""
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import comtypes.client

PNG_SHAPE = 2       # ppShapeFormatPNG
REL_IMG = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
MARGEN_PT = 9       # margen del recorte: las ecuaciones pueden desbordar su caja
EMU_PT = 12700


def transparente(im, tinta=(0x26, 0x28, 0x2A)):
    """Fondo blanco -> transparente: la opacidad sale de la luminancia y el color es la tinta."""
    from PIL import Image
    gris = im.convert("L")
    alfa = gris.point(lambda v: max(0, min(255, round((248 - v) * 255 / (248 - 40)))))
    out = Image.new("RGBA", im.size, tinta + (0,))
    out.putalpha(alfa)
    return out


def exportar_pngs(pptx, carpeta, ancho_px=4000):
    """Exporta cada ecuación a PNG recortándola de la diapositiva renderizada a alta resolución.

    Se ocultan en memoria las demás formas de la diapositiva para que el recorte contenga solo
    la ecuación; la presentación se cierra sin guardar.
    """
    from PIL import Image
    pp = comtypes.client.CreateObject("PowerPoint.Application")
    hechos = {}
    try:
        pres = pp.Presentations.Open(os.path.abspath(pptx), True, False, False)
        ancho_pt = pres.PageSetup.SlideWidth
        alto_px = round(ancho_px * pres.PageSetup.SlideHeight / ancho_pt)
        k = ancho_px / ancho_pt
        for i in range(1, pres.Slides.Count + 1):
            s = pres.Slides(i)
            eqs = [s.Shapes(j) for j in range(1, s.Shapes.Count + 1) if s.Shapes(j).Name.startswith("EQ__")]
            if not eqs:
                continue
            for j in range(1, s.Shapes.Count + 1):
                if not s.Shapes(j).Name.startswith("EQ__"):
                    s.Shapes(j).Visible = 0
            completa = Path(carpeta) / f"diapositiva_{i:02d}.png"
            s.Export(str(completa.resolve()), "PNG", ancho_px, alto_px)
            im = Image.open(completa).convert("RGB")
            for sh in eqs:
                m = MARGEN_PT
                caja = (round((sh.Left - m) * k), round((sh.Top - m) * k), round((sh.Left + sh.Width + m) * k), round((sh.Top + sh.Height + m) * k))
                ruta = Path(carpeta) / f"ecuacion_{i:02d}_{sh.Name.split('__')[1]}.png"
                transparente(im.crop(caja)).save(ruta, optimize=True)
                hechos[(i, sh.Name)] = ruta
        pres.Close()
    finally:
        pp.Quit()
    return hechos


def main(entrada, salida):
    tmp = Path(tempfile.mkdtemp(prefix="ecuaciones_"))
    pngs = exportar_pngs(entrada, tmp)
    zin = zipfile.ZipFile(entrada)
    nombres = set(zin.namelist())
    ct = zin.read("[Content_Types].xml").decode("utf-8")
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Default ", '<Default Extension="png" ContentType="image/png"/><Default ', 1)
    nuevos, rels_mod, slides_mod = {}, {}, {}
    for (i, nombre), ruta in pngs.items():
        sl = f"ppt/slides/slide{i}.xml"
        rl = f"ppt/slides/_rels/slide{i}.xml.rels"
        xml = slides_mod.get(sl) or zin.read(sl).decode("utf-8")
        rels = rels_mod.get(rl) or zin.read(rl).decode("utf-8")
        rid = f"rIdEq{len(re.findall('rIdEq', rels)) + 1}"
        media = f"ppt/media/{ruta.name}"
        assert media not in nombres
        nuevos[media] = ruta.read_bytes()
        rels = rels.replace("</Relationships>", f'<Relationship Id="{rid}" Type="{REL_IMG}" Target="../media/{ruta.name}"/></Relationships>')
        # Respaldo de esta ecuación: misma forma, relleno de imagen y texto vacío.
        patron = re.compile(r'(<mc:Fallback[^>]*>\s*<p:sp>\s*<p:nvSpPr><p:cNvPr id="\d+" name="' + re.escape(nombre) + r'"(?:/>|></p:cNvPr>).*?</p:sp>\s*</mc:Fallback>)', re.S)
        m = patron.search(xml)
        assert m, (i, nombre)
        fb = m.group(1)
        fb2 = re.sub(r"<a:noFill/><a:ln(?:/>|></a:ln>)", f'<a:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></a:blipFill><a:ln><a:noFill/></a:ln>', fb, count=1)
        fb2 = re.sub(r"<a:p>.*</a:p>", '<a:p><a:endParaRPr lang="es-CO"/></a:p>', fb2, count=1, flags=re.S)
        me = MARGEN_PT * EMU_PT
        fb2 = re.sub(r'<a:off x="(\d+)" y="(\d+)"/><a:ext cx="(\d+)" cy="(\d+)"/>',
                     lambda q: (f'<a:off x="{int(q[1]) - me}" y="{int(q[2]) - me}"/>'
                                f'<a:ext cx="{int(q[3]) + 2 * me}" cy="{int(q[4]) + 2 * me}"/>'), fb2, count=1)
        assert fb2 != fb, (i, nombre)
        xml = xml.replace(fb, fb2)
        slides_mod[sl], rels_mod[rl] = xml, rels
    with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            if it.filename == "[Content_Types].xml":
                zout.writestr(it, ct)
            elif it.filename in slides_mod:
                zout.writestr(it, slides_mod[it.filename])
            elif it.filename in rels_mod:
                zout.writestr(it, rels_mod[it.filename])
            else:
                zout.writestr(it, zin.read(it.filename))
        for nombre, datos in nuevos.items():
            zout.writestr(nombre, datos)
    print(f"{len(pngs)} ecuaciones con imagen de respaldo")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

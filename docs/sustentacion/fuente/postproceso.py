"""Inserta las ecuaciones OMML y las barras de error en el borrador de pptxgenjs.

    python postproceso.py entrada.pptx salida.pptx

Busca los cuadros de texto llamados EQ__<id>__<pt>__<color>__<alineación> y
reemplaza su contenido por la ecuación nativa (ecuaciones.py). Los gráficos
cuyo título empieza por ERR__<clave> reciben barras de error personalizadas
tomadas de errores.json, y ese prefijo se elimina del título.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import ecuaciones

AQUI = Path(__file__).resolve().parent
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
A14 = "http://schemas.microsoft.com/office/drawing/2010/main"

RE_SP = re.compile(r"<p:sp>(?:(?!<p:sp>).)*?</p:sp>", re.S)


def ecuaciones_en(xml):
    usadas = []

    def cambiar(m):
        sp = m.group(0)
        nombre = re.search(r'<p:cNvPr id="\d+" name="([^"]+)"', sp)
        if not nombre or not nombre.group(1).startswith("EQ__"):
            return sp
        _, eid, pt, color, alinear = nombre.group(1).split("__")
        usadas.append(eid)
        cuerpo = re.search(r"(<p:txBody>.*?<a:lstStyle/>)(.*)(</p:txBody>)", sp, re.S)
        assert cuerpo, eid
        nuevo = sp[:cuerpo.start(2)] + ecuaciones.parrafo(eid, float(pt), color, alinear) + sp[cuerpo.end(2):]
        return (f'<mc:AlternateContent xmlns:mc="{MC}"><mc:Choice xmlns:a14="{A14}" Requires="a14">'
                f"{nuevo}</mc:Choice><mc:Fallback>{sp}</mc:Fallback></mc:AlternateContent>")

    return RE_SP.sub(cambiar, xml), usadas


def num_lit(vals):
    pts = "".join(f'<c:pt idx="{i}"><c:v>{v:.6f}</c:v></c:pt>' for i, v in enumerate(vals))
    return f'<c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="{len(vals)}"/>{pts}</c:numLit>'


def barras_en(xml, errores):
    m = re.search(r"<a:t>ERR__([A-Za-z0-9_]+)</a:t>", xml)
    if not m:
        return xml, None
    clave = m.group(1)
    xml = xml.replace(m.group(0), "<a:t></a:t>")
    # El título solo existía para identificar el gráfico: se elimina.
    xml = re.sub(r"<c:title>.*?</c:title>", "", xml, count=1, flags=re.S)
    xml = xml.replace('<c:autoTitleDeleted val="0"/>', '<c:autoTitleDeleted val="1"/>')
    series = errores[clave]
    partes = re.split(r"(<c:ser>.*?</c:ser>)", xml, flags=re.S)
    k = 0
    for i, parte in enumerate(partes):
        if not parte.startswith("<c:ser>"):
            continue
        s = series[k]
        k += 1
        barra = ('<c:errBars><c:errBarType val="both"/><c:errValType val="cust"/><c:noEndCap val="0"/>'
                 f"<c:plus>{num_lit(s['mas'])}</c:plus><c:minus>{num_lit(s['menos'])}</c:minus>"
                 f'<c:spPr><a:ln w="15875"><a:solidFill><a:srgbClr val="{s["color"]}"/></a:solidFill></a:ln></c:spPr>'
                 "</c:errBars>")
        partes[i] = parte.replace("<c:cat>", barra + "<c:cat>", 1)
    assert k == len(series), (clave, k)
    return "".join(partes), clave


def main(entrada, salida):
    errores = json.loads((AQUI / "errores.json").read_text(encoding="utf-8")) if (AQUI / "errores.json").exists() else {}
    zin = zipfile.ZipFile(entrada)
    todas, graficos = [], []
    with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            datos = zin.read(item.filename)
            if re.match(r"ppt/slides/slide\d+\.xml$", item.filename):
                xml, usadas = ecuaciones_en(datos.decode("utf-8"))
                todas += usadas
                datos = xml.encode("utf-8")
            elif re.match(r"ppt/charts/chart\d+\.xml$", item.filename):
                xml, clave = barras_en(datos.decode("utf-8"), errores)
                if clave:
                    graficos.append(clave)
                datos = xml.encode("utf-8")
            zout.writestr(item, datos)
    print(f"{len(todas)} ecuaciones insertadas; barras de error en: {graficos or 'ninguno'}")
    faltan = set(ecuaciones.ECUACIONES) - set(todas)
    if faltan:
        print("Ecuaciones definidas sin usar:", sorted(faltan))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

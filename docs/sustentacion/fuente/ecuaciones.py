"""Ecuaciones de la presentación como OMML nativo de PowerPoint.

Cada ecuación se escribe con un pequeño constructor y se inserta después en el
cuadro de texto que construir.js reserva con el nombre EQ__<id>. Al abrir la
presentación en PowerPoint las ecuaciones se editan con el editor de ecuaciones.
La notación sigue el capítulo 3 de la tesis.
"""
from xml.sax.saxutils import escape

FUENTE = ('<a:latin typeface="Cambria Math" panose="02040503050406030204" pitchFamily="18" charset="0"/>'
          '<a:cs typeface="Cambria Math" panose="02040503050406030204" pitchFamily="18" charset="0"/>')


class Ctx:
    def __init__(self, sz=20, color="161A20"):
        self.sz = sz
        self.color = color

    def rpr(self, i=True, b=False, color=None):
        c = color or self.color
        return (f'<a:rPr lang="es-CO" sz="{int(self.sz * 100)}" b="{int(b)}" i="{int(i)}">'
                f'<a:solidFill><a:srgbClr val="{c}"/></a:solidFill>{FUENTE}</a:rPr>')

    def ctrl(self):
        return f"<m:ctrlPr>{self.rpr()}</m:ctrlPr>"


def _mapa(c, negrita):
    o = ord(c)
    if "a" <= c <= "z":
        if not negrita and c == "h":
            return "ℎ"
        return chr((0x1D482 if negrita else 0x1D44E) + o - ord("a"))
    if "A" <= c <= "Z":
        return chr((0x1D468 if negrita else 0x1D434) + o - ord("A"))
    if 0x3B1 <= o <= 0x3C9:
        return chr((0x1D736 if negrita else 0x1D6FC) + o - 0x3B1)
    return c


def _seq(*partes):
    def f(ctx):
        return "".join(p(ctx) if callable(p) else I(p)(ctx) for p in partes)
    return f


def I(texto, color=None):
    """Letras en itálica matemática (variables escalares)."""
    t = "".join(_mapa(c, False) for c in texto)
    return lambda ctx: f'<m:r>{ctx.rpr(i=True, color=color)}<m:t>{escape(t)}</m:t></m:r>'


def B(texto, color=None):
    """Negrita itálica (vectores)."""
    t = "".join(_mapa(c, True) for c in texto)
    return lambda ctx: f'<m:r>{ctx.rpr(i=True, b=True, color=color)}<m:t>{escape(t)}</m:t></m:r>'


OPERADORES = {"=", "+", "−", "·", "∝", "≤", ">", "≥", "⟹", "←", "→",
              "×", "|", ",", ":"}


def T(texto, color=None):
    """Texto recto: números, operadores, nombres de función y palabras.

    Los operadores aislados se escriben sin espacios: el motor de ecuaciones de
    PowerPoint ya aplica el espaciado matemático.
    """
    if texto.strip() in OPERADORES:
        texto = texto.strip()
    return lambda ctx: (f'<m:r><m:rPr><m:sty m:val="p"/></m:rPr>{ctx.rpr(i=False, color=color)}'
                        f'<m:t xml:space="preserve">{escape(texto)}</m:t></m:r>')


def U(texto, color=None):
    """Carácter Unicode ya formado (por ejemplo 𝚺, 𝐈, 𝟙, 𝔼), sin estilo adicional."""
    return lambda ctx: (f'<m:r><m:rPr><m:sty m:val="p"/></m:rPr>{ctx.rpr(i=False, color=color)}'
                        f'<m:t>{escape(texto)}</m:t></m:r>')


def S(*p):
    return _seq(*p)


def frac(num, den):
    return lambda ctx: (f"<m:f><m:fPr>{ctx.ctrl()}</m:fPr><m:num>{S(num)(ctx)}</m:num>"
                        f"<m:den>{S(den)(ctx)}</m:den></m:f>")


def sub(base, s):
    return lambda ctx: (f"<m:sSub><m:sSubPr>{ctx.ctrl()}</m:sSubPr><m:e>{S(base)(ctx)}</m:e>"
                        f"<m:sub>{S(s)(ctx)}</m:sub></m:sSub>")


def sup(base, s):
    return lambda ctx: (f"<m:sSup><m:sSupPr>{ctx.ctrl()}</m:sSupPr><m:e>{S(base)(ctx)}</m:e>"
                        f"<m:sup>{S(s)(ctx)}</m:sup></m:sSup>")


def subsup(base, s, p):
    return lambda ctx: (f"<m:sSubSup><m:sSubSupPr>{ctx.ctrl()}</m:sSubSupPr><m:e>{S(base)(ctx)}</m:e>"
                        f"<m:sub>{S(s)(ctx)}</m:sub><m:sup>{S(p)(ctx)}</m:sup></m:sSubSup>")


def hat(base):
    return lambda ctx: (f'<m:acc><m:accPr><m:chr m:val="̂"/>{ctx.ctrl()}</m:accPr>'
                        f"<m:e>{S(base)(ctx)}</m:e></m:acc>")


def tilde(base):
    return lambda ctx: (f'<m:acc><m:accPr><m:chr m:val="̃"/>{ctx.ctrl()}</m:accPr>'
                        f"<m:e>{S(base)(ctx)}</m:e></m:acc>")


def d(contenido, ini="(", fin=")"):
    return lambda ctx: (f'<m:d><m:dPr><m:begChr m:val="{escape(ini)}"/><m:endChr m:val="{escape(fin)}"/>'
                        f"{ctx.ctrl()}</m:dPr><m:e>{S(contenido)(ctx)}</m:e></m:d>")


def norma(x):
    return d(x, "‖", "‖")


def raiz(x):
    return lambda ctx: (f'<m:rad><m:radPr><m:degHide m:val="1"/>{ctx.ctrl()}</m:radPr><m:deg/>'
                        f"<m:e>{S(x)(ctx)}</m:e></m:rad>")


def suma(abajo, arriba, cuerpo, chr_="∑"):
    def f(ctx):
        oculta = '<m:supHide m:val="1"/>' if arriba is None else ""
        sup_ = "" if arriba is None else S(arriba)(ctx)
        return (f'<m:nary><m:naryPr><m:chr m:val="{chr_}"/><m:limLoc m:val="undOvr"/>{oculta}{ctx.ctrl()}'
                f"</m:naryPr><m:sub>{S(abajo)(ctx)}</m:sub><m:sup>{sup_}</m:sup><m:e>{S(cuerpo)(ctx)}</m:e></m:nary>")
    return f


def casos(*filas):
    def f(ctx):
        cuerpo = "".join(f"<m:e>{S(fl)(ctx)}</m:e>" for fl in filas)
        return (f'<m:d><m:dPr><m:begChr m:val="{{"/><m:endChr m:val=""/>{ctx.ctrl()}</m:dPr>'
                f"<m:e><m:eqArr><m:eqArrPr>{ctx.ctrl()}</m:eqArrPr>{cuerpo}</m:eqArr></m:e></m:d>")
    return f


# ---------------------------------------------------------------- piezas de notación
L = d(I("l"))           # (l)
ESP = T(" ")       # espacio largo entre expresiones de una misma línea
ESP2 = T("  ")


def x(i="i", capa=L):
    return subsup(B("x"), I(i), capa)


def rhat(capa=L):
    return sup(hat(B("r")), capa)


def DP(rama):
    return S(T("Δ"), sub(I("p"), I(rama)))


ECUACIONES = {
    # --- diapositiva: flujo residual y lectura
    "residual": S(x(), T(" = "), x(capa=d(S(I("l"), T("−"), T("1")))), T(" + "),
                  subsup(B("a"), I("i"), L), T(" + "), subsup(B("m"), I("i"), L)),
    "desplegado": S(x(), T(" = "), x(capa=d(T("0"))), T(" + "),
                    suma(S(I("k"), T("≤"), I("l")), None,
                         d(S(subsup(B("a"), I("i"), d(I("k"))), T(" + "), subsup(B("m"), I("i"), d(I("k"))))))),
    "mascara": S(sub(I("α"), I("ij")), T(" = "), sub(T("softmax"), I("j")),
                 d(S(frac(S(sub(B("q"), I("i")), T("·"), sub(B("k"), I("j"))), raiz(sub(I("d"), I("k")))),
                     T(" + "), sub(I("M"), I("ij")))),
                 ESP, sub(I("M"), I("ij")), T(" = "),
                 casos(S(T("0,"), T(" "), I("j"), T(" ≤ "), I("i")),
                       S(T("−∞,"), T(" "), I("j"), T(" > "), I("i")))),
    "salida": S(I("P"), d(S(sub(I("t"), S(I("n"), T("+1"))), T(" | "), sub(I("t"), T("1")), T(", …, "),
                            sub(I("t"), I("n")))),
                T(" = "), T("softmax"), d(S(sub(U("\U0001D416"), I("U")), T(" RMSNorm"),
                                           d(subsup(B("x"), I("n"), d(I("L"))))))),
    # --- dirección y proyección
    "direccion": S(sup(B("r"), L), T(" = "), subsup(B("μ"), I("h"), L), T(" − "),
                   subsup(B("μ"), I("b"), L), T(" = "), frac(T("1"), I("n")),
                   suma(S(I("i"), T("=1")), I("n"),
                        d(S(subsup(B("x"), S(I("h"), T(","), I("i")), L), T(" − "),
                            subsup(B("x"), S(I("b"), T(","), I("i")), L))))),
    "unitaria": S(rhat(), T(" = "), frac(sup(B("r"), L), norma(sup(B("r"), L)))),
    "proyeccion": S(sup(I("p"), L), d(B("x")), T(" = "), sup(B("x"), L), T(" · "), rhat(), T(" = "),
                    norma(sup(B("x"), L)), T(" cos "), I("θ")),
    "fisher": S(sub(B("w"), T("Fisher")), T(" ∝ "), sup(sub(U("\U0001D6BA"), I("W")), T("−1")),
                d(S(sub(B("μ"), I("h")), T(" − "), sub(B("μ"), I("b"))))),
    "ridge": S(frac(S(sup(d(S(sub(hat(U("\U0001D6BA")), I("W")), T(" + "), I("λ"), U("\U0001D408"))), T("−1")),
                      d(S(sub(B("μ"), I("h")), T(" − "), sub(B("μ"), I("b"))))),
                    norma(T("·"))),
               T("  →  "), hat(B("r")), T("   cuando  "), I("λ"), T(" → ∞")),
    # --- desplazamiento y control benigno
    "delta": S(subsup(I("δ"), S(I("h"), T(","), I("i")), L), T(" = "),
               subsup(I("p"), T("chat"), L), d(sub(B("x"), S(I("h"), T(","), I("i")))), T(" − "),
               subsup(I("p"), T("ag"), L), d(sub(B("x"), S(I("h"), T(","), I("i"))))),
    "gap": S(T("Δ"), subsup(I("p"), I("h"), L), T(" = "), frac(T("1"), I("n")),
             suma(S(I("i"), T("=1")), I("n"), subsup(I("δ"), S(I("h"), T(","), I("i")), L))),
    "interaccion": S(sup(T("Δ"), T("2")), I("p"), T(" = "), DP("h"), T(" − "), DP("b"),
                     ESP2, I("φ"), T(" = "), frac(DP("b"), DP("h"))),
    "prueba_t": S(I("t"), T(" = "), frac(DP("h"), S(sub(I("s"), I("δ")), T(" / "), raiz(I("n")))),
                  ESP, sub(I("d"), I("z")), T(" = "), frac(DP("h"), sub(I("s"), I("δ"))),
                  ESP, I("t"), T(" = "), sub(I("d"), I("z")), raiz(I("n"))),
    # --- escala y orden
    "estandar": S(I("d"), T(" = "), frac(I("S"), I("s")), ESP2, I("κ"), T(" = "),
                  frac(sub(I("s"), T("ag")), sub(I("s"), T("chat")))),
    "afin": S(sub(I("p"), T("ag")), T(" = "), I("a"), T(" "), sub(I("p"), T("chat")), T(" + "), I("b"),
              T(","), ESP, I("a"), T(" > "), T("0")),
    "afin2": S(T("⟹"), ESP, sub(I("S"), T("ag")), T(" = "), I("a"), T(" "), sub(I("S"), T("chat")), T(","), ESP,
               sub(I("d"), T("ag")), T(" = "), sub(I("d"), T("chat")), T(","), ESP, I("κ"), T(" = "), I("a")),
    "auroc": S(hat(T("AUROC")), T(" = "), frac(T("1"), S(sub(I("n"), I("h")), T(" "), sub(I("n"), I("b")))),
               suma(I("i"), None, suma(I("j"), None,
                    d(S(U("\U0001D7D9"), d(S(sub(I("p"), S(I("h"), T(","), I("i"))), T(" > "),
                                             sub(I("p"), S(I("b"), T(","), I("j"))))),
                        T(" + "), frac(T("1"), T("2")), U("\U0001D7D9"),
                        d(S(sub(I("p"), S(I("h"), T(","), I("i"))), T(" = "),
                            sub(I("p"), S(I("b"), T(","), I("j")))))), "[", "]")))),
    # --- validación
    "sesgo": S(U("\U0001D53C"), d(sup(norma(sup(B("r"), L)), T("2")), "[", "]"), T(" = "),
               sup(norma(B("r")), T("2")), T(" + "),
               frac(S(T("tr"), d(sub(U("\U0001D6BA"), T("Δ")))), I("n"))),
    "crossfit": S(sup(tilde(I("p")), L), d(sub(B("x"), I("i"))), T(" = "), subsup(B("x"), I("i"), L), T(" · "),
                  subsup(hat(B("r")), S(T("−"), I("k"), d(I("i"))), L)),
    "permutacion": S(sup(I("T"), d(I("b"))), T(" = "), frac(T("1"), I("n")),
                     suma(S(I("i"), T("=1")), I("n"), S(subsup(I("ε"), I("i"), d(I("b"))), sub(I("δ"), I("i")))),
                     ESP2, I("p"), T(" = "),
                     frac(S(T("1 + #"), d(S(I("b"), T(" : "), sup(I("T"), d(I("b"))), T(" ≥ "), T("Δ"), I("p")),
                                         "{", "}")), S(I("B"), T(" + 1")))),
    "signos": S(I("p"), T(" = "), suma(S(I("j"), T(" ≥ "), I("k")), None,
                                       S(d(frac(I("n"), I("j"))), sup(T("2"), S(T("−"), I("n")))))),
    "nadeau": S(sub(hat(T("Var")), T("corr")), T(" = "), d(S(frac(T("1"), I("J")), T(" + "),
                                                         frac(sub(I("n"), T("eval")), sub(I("n"), T("ent"))))),
                T(" "), sup(hat(I("σ")), T("2"))),
    # --- reorientación
    "coseno": S(T("cos "), I("θ"), T(" = "), hat(B("r")), T(" · "), sub(hat(B("w")), T("DM")),
                ESP2, sub(T("cos"), T("norm")), T(" = "),
                frac(sub(T("cos"), T("×")), raiz(S(sub(I("ρ"), T("chat")), T(" "), sub(I("ρ"), T("ag")))))),
    # --- intervención
    "adicion": S(sup(B("x"), d(T("56"))), T("←"), sup(B("x"), d(T("56"))), T(" + "), I("α"), T(" "),
                 sup(hat(B("r")), d(T("56"))), ESP2, I("α"), T(" = "), T("dosis"), T(" × "),
                 T("Δ"), subsup(I("p"), I("h"), d(T("56")))),
}

# Expresión lineal de respaldo, usada solo en el contenido alternativo del archivo.
LINEAL = {k: k for k in ECUACIONES}


def parrafo(eid, sz=20, color="161A20", alinear="left"):
    ctx = Ctx(sz, color)
    cuerpo = ECUACIONES[eid](ctx)
    return (f'<a:p><a:pPr algn="{"l" if alinear == "left" else "ctr"}"/><a14:m>'
            f'<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            f'<m:oMathParaPr><m:jc m:val="{alinear}"/></m:oMathParaPr><m:oMath>{cuerpo}</m:oMath></m:oMathPara>'
            f'</a14:m><a:endParaRPr lang="es-CO" sz="{int(sz * 100)}"/></a:p>')

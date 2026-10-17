"""Miniaturas del PDF final para que el guion muestre exactamente su exportación."""
import sys
from pathlib import Path

import pymupdf


def main(pdf, carpeta):
    destino = Path(carpeta)
    destino.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf) as doc:
        for i, pagina in enumerate(doc):
            escala = 1600 / pagina.rect.width
            pagina.get_pixmap(matrix=pymupdf.Matrix(escala, escala)).save(destino / f"s{i+1:02d}.png")
        print(f"Miniaturas: {len(doc)} diapositivas")


if __name__ == "__main__":
    main(*sys.argv[1:3])

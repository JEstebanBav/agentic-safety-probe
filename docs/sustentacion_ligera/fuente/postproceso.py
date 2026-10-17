"""Dos ecuaciones breves y editables, con el mismo motor OMML de la versión completa."""
import sys
from pathlib import Path

COMUN = Path(__file__).resolve().parents[2] / "sustentacion/fuente"
sys.path.insert(0, str(COMUN))
import ecuaciones as e
import postproceso as motor

e.ECUACIONES = {
    "ligera_r": e.S(e.B("r"), e.T("="), e.sub(e.B("μ"), e.T("D")), e.T("−"), e.sub(e.B("μ"), e.T("B"))),
    "ligera_p": e.S(e.I("p"), e.T("="), e.B("x"), e.T("·"), e.hat(e.B("r"))),
}

if __name__ == "__main__":
    motor.main(sys.argv[1], sys.argv[2])

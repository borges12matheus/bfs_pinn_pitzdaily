import re
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Utilidades
# -------------------------------------------------

def latest_dir(base: Path) -> Path:
    """Retorna o diretório do último 'tempo' (ex: 281)."""
    times = sorted(
        [p for p in base.iterdir() if p.is_dir()],
        key=lambda p: float(p.name)
    )
    return times[-1]


def read_grid_file(path: Path) -> pd.DataFrame:
    """
    Lê o arquivo grid2D_p_U.xy no formato:
    x y z p Ux Uy Uz
    """
    rows = []
    num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            nums = num_re.findall(line)
            if len(nums) < 7:
                continue

            x, y, z, p, ux, uy, uz = map(float, nums[:7])
            rows.append((x, y, z, p, ux, uy, uz))

    return pd.DataFrame(
        rows, columns=["x", "y", "z", "p", "Ux", "Uy", "Uz"]
    )


def load_case(case_dir: str) -> pd.DataFrame:
    """
    Carrega o grid2D_p_U.xy do último tempo do caso.
    """
    case = Path(case_dir)
    base = case / "postProcessing" / "sampleDict"
    tdir = latest_dir(base)

    fpath = tdir / "grid2D_p_U.xy"
    if not fpath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {fpath}")

    return read_grid_file(fpath)

# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print("Lendo caso coarse...")
    dfc = load_case("pitzDaily_coarse")

    print("Lendo caso fine...")
    dff = load_case("pitzDaily_fine")

    # Merge ponto a ponto (x,y)
    df = dfc.merge(
        dff[["x", "y", "p", "Ux", "Uy"]].rename(
            columns={
                "p": "p_f",
                "Ux": "Ux_f",
                "Uy": "Uy_f"
            }
        ),
        on=["x", "y"],
        how="inner"
    )

    # Erro coarse -> fine (targets do ML)
    df["dUx"] = df["Ux_f"] - df["Ux"]
    df["dUy"] = df["Uy_f"] - df["Uy"]
    df["dp"]  = df["p_f"]  - df["p"]

    # Reynolds (fixo neste experimento)
    df["Re"] = 25400.0

    # Dataset final
    out = df[[
        "x", "y",
        "Ux", "Uy", "p",
        "Re",
        "dUx", "dUy", "dp"
    ]].copy()

    out.to_parquet("dataset_pitzDaily.parquet", index=False)
    print("Dataset salvo em: dataset_pitzDaily.parquet")
    print("Total de amostras:", len(out))


if __name__ == "__main__":
    main()

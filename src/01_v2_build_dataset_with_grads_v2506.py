import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neighbors import NearestNeighbors


# ----------------------------
# Leitura dos .xy
# ----------------------------
def latest_dir(base: Path) -> Path:
    times = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: float(p.name))
    return times[-1]

def read_grid_file(path: Path) -> pd.DataFrame:
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
    return pd.DataFrame(rows, columns=["x", "y", "z", "p", "Ux", "Uy", "Uz"])

def load_case(case_dir: str) -> pd.DataFrame:
    case = Path(case_dir)
    base = case / "postProcessing" / "sampleDict"
    tdir = latest_dir(base)
    fpath = tdir / "grid2D_p_U.xy"
    if not fpath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {fpath}")
    return read_grid_file(fpath)


# ----------------------------
# Gradiente por kNN + mínimos quadrados locais
# ----------------------------
def local_gradients_knn(xy: np.ndarray, u: np.ndarray, k: int = 12, eps: float = 1e-12):
    """
    Estima dU/dx e dU/dy em cada ponto via ajuste local:
      u ≈ a + b*(x-xi) + c*(y-yi)
    usando k vizinhos mais próximos (inclui o próprio ponto).
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xy)
    dists, idxs = nbrs.kneighbors(xy)

    dudx = np.zeros(len(u), dtype=float)
    dudy = np.zeros(len(u), dtype=float)

    for i in range(len(u)):
        neigh = idxs[i]
        xi, yi = xy[i]

        X = xy[neigh, 0] - xi
        Y = xy[neigh, 1] - yi
        Z = u[neigh] - u[i]  # tira offset para estabilizar

        # matriz do ajuste: Z ≈ b*X + c*Y  (a ~ 0 por causa do offset)
        A = np.column_stack([X, Y])

        # pesos por distância (opcional, melhora perto de buracos)
        w = 1.0 / (dists[i] + eps)
        W = np.diag(w)

        # resolve (A^T W A) [b,c] = A^T W Z
        ATA = A.T @ W @ A
        ATZ = A.T @ W @ Z

        try:
            bc = np.linalg.solve(ATA, ATZ)
        except np.linalg.LinAlgError:
            # fallback se der singular: mínimos quadrados simples
            bc, *_ = np.linalg.lstsq(A, Z, rcond=None)

        dudx[i] = bc[0]
        dudy[i] = bc[1]

    return dudx, dudy


def main():
    print("Lendo coarse...")
    dfc = load_case("pitzDaily_coarse")
    print("Lendo fine...")
    dff = load_case("pitzDaily_fine")

    # gradientes no COARSE (sem assumir grade estruturada)
    xy = dfc[["x", "y"]].values.astype(np.float64)

    print("Calculando gradientes kNN (coarse)...")
    dUx_dx, dUx_dy = local_gradients_knn(xy, dfc["Ux"].values.astype(np.float64), k=12)
    dUy_dx, dUy_dy = local_gradients_knn(xy, dfc["Uy"].values.astype(np.float64), k=12)

    dfc = dfc.copy()
    dfc["dUx_dx"] = dUx_dx
    dfc["dUx_dy"] = dUx_dy
    dfc["dUy_dx"] = dUy_dx
    dfc["dUy_dy"] = dUy_dy

    # merge coarse/fine por (x,y)
    df = dfc.merge(
        dff[["x", "y", "p", "Ux", "Uy"]].rename(
            columns={"p": "p_f", "Ux": "Ux_f", "Uy": "Uy_f"}
        ),
        on=["x", "y"],
        how="inner",
    )

    # targets
    df["dUx"] = df["Ux_f"] - df["Ux"]
    df["dUy"] = df["Uy_f"] - df["Uy"]
    df["dp"]  = df["p_f"]  - df["p"]

    # Re (ajuste se quiser ler do case; aqui fixo)
    df["Re"] = 25400.0

    out = df[[
        "x", "y",
        "Ux", "Uy", "p",
        "dUx_dx", "dUx_dy", "dUy_dx", "dUy_dy",
        "Re",
        "dUx", "dUy", "dp"
    ]].copy()

    out.to_parquet("dataset_pitzDaily_with_grads.parquet", index=False)
    print("Salvo: dataset_pitzDaily_with_grads.parquet")
    print("Linhas:", len(out))


if __name__ == "__main__":
    main()

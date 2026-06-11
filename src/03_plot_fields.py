# 03_plot_fields.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# util: plot de campo 2D
# ----------------------------
def plot_field(ax, df, val, title, cmap="viridis"):
    sc = ax.tricontourf(df["x"], df["y"], df[val], levels=60, cmap=cmap)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return sc

def main():
    df = pd.read_parquet("predictions_test.parquet")

    # campos
    df["Ux_f"] = df["Ux"] + df["dUx"]          # fine
    df["Ux_corr"] = df["Ux_corr"]               # já salvo no treino

    # limites comuns (pra comparação justa)
    vmin = min(df["Ux"].min(), df["Ux_f"].min(), df["Ux_corr"].min())
    vmax = max(df["Ux"].max(), df["Ux_f"].max(), df["Ux_corr"].max())

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    sc0 = axs[0].tricontourf(df["x"], df["y"], df["Ux"], levels=60, vmin=vmin, vmax=vmax)
    axs[0].set_title("Ux — Coarse")
    axs[0].set_aspect("equal", adjustable="box")

    sc1 = axs[1].tricontourf(df["x"], df["y"], df["Ux_f"], levels=60, vmin=vmin, vmax=vmax)
    axs[1].set_title("Ux — Fine (referência)")
    axs[1].set_aspect("equal", adjustable="box")

    sc2 = axs[2].tricontourf(df["x"], df["y"], df["Ux_corr"], levels=60, vmin=vmin, vmax=vmax)
    axs[2].set_title("Ux — Corrigido (ML)")
    axs[2].set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc2, ax=axs, shrink=0.9)
    cbar.set_label("Ux [m/s]")

    for ax in axs:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    plt.savefig("Ux_coarse_fine_corrected.png", dpi=200)
    plt.show()
    print("Salvo: Ux_coarse_fine_corrected.png")

    # --------- mapa de erro (opcional, muito bom p/ análise)
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    err_coarse = np.sqrt((df["Ux_f"] - df["Ux"])**2 + (df["Uy"] + df["dUy"] - df["Uy"])**2)
    err_corr   = np.sqrt((df["Ux_f"] - df["Ux_corr"])**2 + (df["Uy"] + df["dUy"] - df["Uy"])**2)

    sc3 = ax2[0].tricontourf(df["x"], df["y"], err_coarse, levels=60, cmap="magma")
    ax2[0].set_title("Erro |U| — Coarse vs Fine")
    ax2[0].set_aspect("equal", adjustable="box")

    sc4 = ax2[1].tricontourf(df["x"], df["y"], err_corr, levels=60, cmap="magma")
    ax2[1].set_title("Erro |U| — Corrigido vs Fine")
    ax2[1].set_aspect("equal", adjustable="box")

    cbar2 = fig2.colorbar(sc4, ax=ax2, shrink=0.9)
    cbar2.set_label("|Erro de velocidade|")

    for ax in ax2:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    plt.savefig("Erro_coarse_vs_corrected.png", dpi=200)
    plt.show()
    print("Salvo: Erro_coarse_vs_corrected.png")

if __name__ == "__main__":
    main()

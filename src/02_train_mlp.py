# 02_train_mlp.py
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ----------------------------
# Modelo
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Métricas
# ----------------------------
def mae_vec(u1, v1, u2, v2) -> float:
    """MAE vetorial médio: E[ sqrt((du)^2 + (dv)^2) ]"""
    return float(np.mean(np.sqrt((u1 - u2) ** 2 + (v1 - v2) ** 2)))


def rer(u_corr, v_corr, u_c, v_c, u_f, v_f) -> float:
    """
    RER = 1 - ||Ucorr - Ufine|| / ||Ucoarse - Ufine||
    norma L2 global.
    """
    num = np.sqrt(np.sum((u_corr - u_f) ** 2 + (v_corr - v_f) ** 2))
    den = np.sqrt(np.sum((u_c - u_f) ** 2 + (v_c - v_f) ** 2))
    return float(1.0 - (num / den))



# ----------------------------
# Treino
# ----------------------------
def main():
    # original simple one:
    #df = pd.read_parquet("dataset_pitzDaily.parquet")
    #feat_cols = ["x", "y", "Ux", "Uy", "p", "Re"]
    
    # this one with grads:
    # df = pd.read_parquet("dataset_pitzDaily_with_grads.parquet")
    # feat_cols = ["x", "y", "Ux", "Uy", "p", "dUx_dx", "dUx_dy", "dUy_dx", "dUy_dy", "Re"]
    
    # this one with vorticity:
    df = pd.read_parquet("dataset_pitzDaily_with_vorticity.parquet")
    feat_cols = [
    "x","y","Ux","Uy","p",
    "Um","wz",
    "dUx_dx","dUx_dy","dUy_dx","dUy_dy",
    "Re"
    ]

    targ_cols = ["dUx", "dUy"]

    X = df[feat_cols].values.astype(np.float32)
    Y = df[targ_cols].values.astype(np.float32)

    # guardamos indices pra puxar coarse/fine depois com consistência
    idx_all = np.arange(len(df))

    # 1) split: train vs temp (val+test)
    X_tr, X_tmp, Y_tr, Y_tmp, idx_tr, idx_tmp = train_test_split(
        X, Y, idx_all, test_size=0.30, random_state=42, shuffle=True
    )

    # 2) split: val vs test (metade/metade do tmp => 15%/15%)
    X_va, X_te, Y_va, Y_te, idx_va, idx_te = train_test_split(
        X_tmp, Y_tmp, idx_tmp, test_size=0.50, random_state=42, shuffle=True
    )

    # normalização (fit só no treino)
    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(Y_tr)

    X_tr_s = sx.transform(X_tr).astype(np.float32)
    X_va_s = sx.transform(X_va).astype(np.float32)
    X_te_s = sx.transform(X_te).astype(np.float32)

    Y_tr_s = sy.transform(Y_tr).astype(np.float32)
    Y_va_s = sy.transform(Y_va).astype(np.float32)
    Y_te_s = sy.transform(Y_te).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_t = torch.tensor(X_tr_s, device=device)
    Ytr_t = torch.tensor(Y_tr_s, device=device)
    Xva_t = torch.tensor(X_va_s, device=device)
    Yva_t = torch.tensor(Y_va_s, device=device)

    model = MLP(in_dim=X.shape[1], hidden=64, out_dim=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    # treino + early stopping na VAL
    best_val = float("inf")
    patience = 12
    bad = 0
    max_epochs = 300

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = loss_fn(pred, Ytr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), Yva_t).item()

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), "mlp_dU.pt")
        else:
            bad += 1

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train={loss.item():.6e} | val={val_loss:.6e}")

        if bad >= patience:
            print("Early stopping (val não melhorou).")
            break

    # salvar scalers
    joblib.dump(sx, "scaler_X.pkl")
    joblib.dump(sy, "scaler_Y.pkl")
    print("Salvo: mlp_dU.pt, scaler_X.pkl, scaler_Y.pkl")

    # ----------------------------
    # Avaliação no TESTE
    # ----------------------------
    model.load_state_dict(torch.load("mlp_dU.pt", map_location=device))
    model.eval()

    Xte_t = torch.tensor(X_te_s, device=device)
    with torch.no_grad():
        d_pred_s = model(Xte_t).cpu().numpy()

    d_pred = sy.inverse_transform(d_pred_s).astype(np.float64)

    # campos coarse/fine do TESTE (usando idx_te)
    u_c = df.loc[idx_te, "Ux"].values.astype(np.float64)
    v_c = df.loc[idx_te, "Uy"].values.astype(np.float64)
    u_f = (df.loc[idx_te, "Ux"].values + df.loc[idx_te, "dUx"].values).astype(np.float64)
    v_f = (df.loc[idx_te, "Uy"].values + df.loc[idx_te, "dUy"].values).astype(np.float64)

    u_corr = u_c + d_pred[:, 0]
    v_corr = v_c + d_pred[:, 1]

    mae_base = mae_vec(u_c, v_c, u_f, v_f)
    mae_corr = mae_vec(u_corr, v_corr, u_f, v_f)
    rer_val = rer(u_corr, v_corr, u_c, v_c, u_f, v_f)
    improve = 100.0 * (1.0 - mae_corr / mae_base) if mae_base > 0 else 0.0

    print("\n--- Métricas (TESTE) ---")
    print(f"MAE vetorial (coarse -> fine): {mae_base:.6f}")
    print(f"MAE vetorial (corrigido -> fine): {mae_corr:.6f}")
    print(f"RER: {rer_val:.4f}")
    print(f"Melhora (MAE): {improve:.2f}%")

    metrics = {
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "n_total": int(len(df)),
        "n_train": int(len(idx_tr)),
        "n_val": int(len(idx_va)),
        "n_test": int(len(idx_te)),
        "mae_base_test": mae_base,
        "mae_corr_test": mae_corr,
        "rer_test": rer_val,
        "improve_mae_pct_test": improve,
        "best_val_loss_scaled": best_val,
        "device": str(device),
    }

    with open("metrics_test.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Salvo: metrics_test.json")

    # salvar previsões do TESTE (pra plotar só o teste)
    out = df.loc[idx_te, ["x", "y", "Ux", "Uy", "p", "dUx", "dUy"]].copy()
    out["dUx_pred"] = d_pred[:, 0]
    out["dUy_pred"] = d_pred[:, 1]
    out["Ux_f"] = out["Ux"] + out["dUx"]
    out["Uy_f"] = out["Uy"] + out["dUy"]
    out["Ux_corr"] = out["Ux"] + out["dUx_pred"]
    out["Uy_corr"] = out["Uy"] + out["dUy_pred"]
    out.to_parquet("predictions_test.parquet", index=False)
    print("Salvo: predictions_test.parquet (somente TESTE)")


if __name__ == "__main__":
    main()

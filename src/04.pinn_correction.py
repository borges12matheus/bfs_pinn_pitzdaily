# pinn_correction.py
# PINN corretor: aprende ΔU (e Δp opcional) para corrigir coarse -> aproximar fine
# Requer no parquet: x,y,Ux,Uy,p,Um,wz,dUx_dx,dUx_dy,dUy_dx,dUy_dy,Re + targets dUx,dUy (se você já tem)
# Ajuste os nomes das colunas de target conforme seu dataset.

import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# -----------------------------
# 1) Dataset
# -----------------------------
FEAT_COLS = [
    "x","y","Ux","Uy","p",
    "Um","wz",
    "dUx_dx","dUx_dy","dUy_dx","dUy_dy",
    "Re"
]

# ajuste aqui se seus targets têm outros nomes
TARGET_COLS = ["dUx", "dUy", "dp"]  # (ΔUx_true, ΔUy_true)
HAS_TARGETS = True           # se quiser rodar sem supervisão, coloque False

class CFDParquet(Dataset):
    def __init__(self, df, feat_cols, target_cols=None, x_scaler=None):
        self.X = df[feat_cols].to_numpy(dtype=np.float32)
        self.has_y = target_cols is not None
        self.Y = df[target_cols].to_numpy(dtype=np.float32) if self.has_y else None

        # scaler só para features (z-score)
        if x_scaler is None:
            mu = self.X.mean(axis=0, keepdims=True)
            sd = self.X.std(axis=0, keepdims=True) + 1e-8
            self.x_mu = mu
            self.x_sd = sd
        else:
            self.x_mu, self.x_sd = x_scaler

        self.Xn = (self.X - self.x_mu) / self.x_sd

    def __len__(self):
        return self.Xn.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.Xn[idx], dtype=DTYPE)
        if self.has_y:
            y = torch.tensor(self.Y[idx], dtype=DTYPE)  # <-- TARGET EM ESCALA FÍSICA
            return x, y
        return x


# -----------------------------
# 2) Modelo
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, depth=5, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        # init leve (ajuda tanh)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 3) Autodiff util
# -----------------------------
def grad(outputs, inputs):
    # outputs: (N,1) or (N,)
    # inputs: (N,1)
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

def laplacian(f, x, y):
    fx = grad(f, x)
    fy = grad(f, y)
    fxx = grad(fx, x)
    fyy = grad(fy, y)
    return fxx + fyy


# -----------------------------
# 4) Resíduos físicos (2D incompressível, estacionário)
# -----------------------------
def pde_residuals(x, y, u_coarse, v_coarse, p_coarse, Re, model_out, predict_dp=True):
    """
    x, y: (N,1) leaf tensors com requires_grad=True (os mesmos do forward)
    model_out:
      se predict_dp=True: (N,3) = [dU, dV, dP]
      senão: (N,2) = [dU, dV]
    """
    if predict_dp:
        dU, dV, dP = model_out[:, [0]], model_out[:, [1]], model_out[:, [2]]
        p_hat = p_coarse + dP
    else:
        dU, dV = model_out[:, [0]], model_out[:, [1]]
        p_hat = p_coarse

    u_hat = u_coarse + dU
    v_hat = v_coarse + dV

    nu = 1.0 / (Re + 1e-12)

    # derivadas
    u_x = grad(u_hat, x); u_y = grad(u_hat, y)
    v_x = grad(v_hat, x); v_y = grad(v_hat, y)

    p_x = grad(p_hat, x); p_y = grad(p_hat, y)

    # laplacianos
    u_lap = laplacian(u_hat, x, y)
    v_lap = laplacian(v_hat, x, y)

    r_cont = u_x + v_y

    adv_u = u_hat * u_x + v_hat * u_y
    adv_v = u_hat * v_x + v_hat * v_y

    r_mom_u = adv_u + p_x - nu * u_lap
    r_mom_v = adv_v + p_y - nu * v_lap

    return r_cont, r_mom_u, r_mom_v


# -----------------------------
# 5) Treino
# -----------------------------
def train_pinn(
    parquet_path,
    batch_data=4096,
    batch_phys=4096,
    epochs_pre=50,
    epochs_phys=100,
    lr=1e-3,
    w_data=1.0,
    w_cont=1.0,
    w_mom=1.0,
    predict_dp=True,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_parquet(parquet_path)

    # Split simples (você pode trocar por split espacial/estratificado por região de wz)
    n = len(df)
    idx = np.random.permutation(n)
    ntr = int(0.8 * n)
    tr_idx, te_idx = idx[:ntr], idx[ntr:]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)

    # Dataset supervisionado (dados)
    train_ds = CFDParquet(df_tr, FEAT_COLS, TARGET_COLS if HAS_TARGETS else None)
    test_ds  = CFDParquet(df_te, FEAT_COLS, TARGET_COLS if HAS_TARGETS else None,
                      x_scaler=(train_ds.x_mu, train_ds.x_sd))


    dl_data = DataLoader(train_ds, batch_size=batch_data, shuffle=True, drop_last=True)

    in_dim = len(FEAT_COLS)
    out_dim = 3 if predict_dp else 2
    net = MLP(in_dim=in_dim, out_dim=out_dim, width=128, depth=5, act=nn.Tanh).to(DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def denorm_y(yhat_n, y_mu, y_sd):
        return yhat_n * torch.tensor(y_sd, device=yhat_n.device) + torch.tensor(y_mu, device=yhat_n.device)

    # helper: pegar colunas específicas do batch (já normalizado) e voltar pra escala original
    # aqui a gente precisa de x,y,Ux,Uy,p,Re em escala original pra PDE.
    feat_index = {c: i for i, c in enumerate(FEAT_COLS)}

    x_mu = torch.as_tensor(train_ds.x_mu, dtype=DTYPE, device=DEVICE)
    x_sd = torch.as_tensor(train_ds.x_sd, dtype=DTYPE, device=DEVICE)

    def unnormalize_X(Xn):
        return Xn * x_sd + x_mu

    # ---- STAGE 1: pré-treino supervisionado (dados)
    if HAS_TARGETS and epochs_pre > 0:
        net.train()
        for ep in range(1, epochs_pre + 1):
            total = 0.0
            for Xn, Yn in dl_data:
                Xn = Xn.to(DEVICE)
                Yn = Yn.to(DEVICE)

                pred = net(Xn)
                loss = torch.mean((pred - Yn) ** 2)   # Yn agora é físico (dUx,dUy[,dp])

                opt.zero_grad()
                loss.backward()
                opt.step()

                total += loss.item()

            if ep % 10 == 0 or ep == 1:
                print(f"[PRE] ep={ep:03d} loss_data={total/len(dl_data):.6e}")

    # ---- STAGE 2: dados + física (REFATORADO / SEM detach no caminho de x,y)
    net.train()
    for ep in range(1, epochs_phys + 1):
        total = 0.0

        for batch in dl_data:
            if HAS_TARGETS:
                Xn, Yn = batch
                Yn = Yn.to(DEVICE)
            else:
                Xn = batch
                Yn = None

            Xn = Xn.to(DEVICE)

            # =========================
            # (1) Loss de dados (ΔU)
            # =========================
            pred_data = net(Xn)  # forward normal
            loss_data = torch.tensor(0.0, device=DEVICE)
            if HAS_TARGETS:
                loss_data = torch.mean((pred_data - Yn) ** 2)   # pred_data é (N,3) e Yn é (N,3)

            # ==========================================
            # (2) Loss física: x,y precisam estar no grafo
            #     então refazemos o forward com x,y requires_grad=True
            # ==========================================
            # Volta para escala física SEM detach (mantém as operações no grafo)
            # valores físicos só pra “montar” o input do PINN (não precisa de grad aqui)
            X_phys = unnormalize_X(Xn).detach() # (N, in_dim) em escala original

            # Cria x,y como tensores do grafo (folhas) para autodiff
            x = X_phys[:, [feat_index["x"]]].clone().requires_grad_(True)
            y = X_phys[:, [feat_index["y"]]].clone().requires_grad_(True)

            # Reinjeta x,y no vetor físico e renormaliza
            X_phys_mod = X_phys.clone()
            X_phys_mod[:, [feat_index["x"]]] = x
            X_phys_mod[:, [feat_index["y"]]] = y

            Xn_mod = (X_phys_mod - x_mu) / x_sd

            # Forward "physics-aware"
            pred_phys = net(Xn_mod)

            # Campos coarse (em escala física) para montar o campo corrigido
            u_c = X_phys[:, [feat_index["Ux"]]]
            v_c = X_phys[:, [feat_index["Uy"]]]
            p_c = X_phys[:, [feat_index["p"]]]
            Re  = X_phys[:, [feat_index["Re"]]]

            #print("x leaf?", x.is_leaf, "req_grad?", x.requires_grad)
            #print("y leaf?", y.is_leaf, "req_grad?", y.requires_grad)


            r_cont, r_mom_u, r_mom_v = pde_residuals(
                x=x, y=y,
                u_coarse=u_c, v_coarse=v_c, p_coarse=p_c,
                Re=Re,
                model_out=pred_phys,
                predict_dp=predict_dp
            )

            loss_cont = (r_cont ** 2).mean()
            huber = torch.nn.SmoothL1Loss(beta=1.0)
            loss_mom = huber(r_mom_u, torch.zeros_like(r_mom_u)) + huber(r_mom_v, torch.zeros_like(r_mom_v))


            loss = w_data * loss_data + w_cont * loss_cont + w_mom * loss_mom

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()

        if ep % 10 == 0 or ep == 1:
            print(
                f"[PINN] ep={ep:03d} "
                f"loss={total/len(dl_data):.6e} "
                f"data={loss_data.item():.3e} cont={loss_cont.item():.3e} mom={loss_mom.item():.3e}"
            )


    return net, (train_ds.x_mu, train_ds.x_sd)
# -----------------------------
# 6) Avaliação / Métricas
# ----------------------------- 

@torch.no_grad()
def evaluate_pinn_physical(
    model, parquet_path, feat_cols, x_scaler,
    batch_size=8192, out_json="metrics_pinn.json"
):
    import json, numpy as np, pandas as pd, torch
    model.eval()
    device = next(model.parameters()).device

    df = pd.read_parquet(parquet_path)

    Ux_c = df["Ux"].to_numpy(np.float32)
    Uy_c = df["Uy"].to_numpy(np.float32)

    dUx_true = df["dUx"].to_numpy(np.float32)
    dUy_true = df["dUy"].to_numpy(np.float32)

    Ux_f = Ux_c + dUx_true
    Uy_f = Uy_c + dUy_true

    X = df[feat_cols].to_numpy(np.float32)
    x_mu, x_sd = x_scaler
    Xn = (X - x_mu) / x_sd

    dUx_pred = np.zeros(len(df), dtype=np.float32)
    dUy_pred = np.zeros(len(df), dtype=np.float32)

    for i0 in range(0, len(df), batch_size):
        i1 = min(i0 + batch_size, len(df))
        xb = torch.tensor(Xn[i0:i1], dtype=torch.float32, device=device)
        out = model(xb)[:, :2].detach().cpu().numpy()
        dUx_pred[i0:i1] = out[:, 0]
        dUy_pred[i0:i1] = out[:, 1]

    Ux_hat = Ux_c + dUx_pred
    Uy_hat = Uy_c + dUy_pred

    def mae_vec(ax, ay, bx, by):
        return float(np.mean(np.sqrt((ax - bx)**2 + (ay - by)**2)))

    mae_coarse = mae_vec(Ux_c, Uy_c, Ux_f, Uy_f)
    mae_corr   = mae_vec(Ux_hat, Uy_hat, Ux_f, Uy_f)
    mae_deltas = mae_vec(dUx_pred, dUy_pred, dUx_true, dUy_true)

    rer = 1.0 - (mae_corr / (mae_coarse + 1e-12))
    melhora_pct = rer * 100.0

    metrics = {
        "mae_vec_coarse_to_fine": mae_coarse,
        "mae_vec_corrected_to_fine": mae_corr,
        "mae_vec_deltas_pred_vs_true": mae_deltas,
        "RER": float(rer),
        "melhora_MAE_pct": float(melhora_pct),
        "N": int(len(df)),
    }

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Métricas (PINN) ---")
    print(f"MAE vetorial (coarse -> fine):      {mae_coarse:.6f}")
    print(f"MAE vetorial (corrigido -> fine):   {mae_corr:.6f}")
    print(f"MAE vetorial (ΔU_pred -> ΔU_true):  {mae_deltas:.6f}")
    print(f"RER:                               {rer:.4f}")
    print(f"Melhora (MAE):                      {melhora_pct:.2f}%")
    print(f"Salvo: {out_json}\n")

    return metrics



if __name__ == "__main__":

    model, xscaler = train_pinn(
        parquet_path="dataset_pitzDaily_with_vorticity.parquet",
        epochs_pre=300,
        epochs_phys=200,
        lr=1e-4,
        w_data=1.0,
        w_cont=0.01,
        w_mom=0.001,
        predict_dp=True
    )

    torch.save({"state_dict": model.state_dict(), "xscaler": xscaler}, "pinn_corrector.pt")

    evaluate_pinn_physical(
        model=model,
        parquet_path="dataset_pitzDaily_with_vorticity.parquet",
        feat_cols=FEAT_COLS,
        x_scaler=xscaler
    )



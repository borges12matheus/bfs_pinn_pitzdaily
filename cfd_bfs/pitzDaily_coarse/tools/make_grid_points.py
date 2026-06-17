import numpy as np
from pathlib import Path

# -----------------------------
# Limites do domínio pitzDaily (m)
# -----------------------------
xmin, xmax = -0.0206, 0.2900
ymin, ymax = -0.0254, 0.0254
z = 0.0

# -----------------------------
# Resolução da grade
# -----------------------------
nx, ny = 400, 160

xs = np.linspace(xmin, xmax, nx)
ys = np.linspace(ymin, ymax, ny)

# -----------------------------
# Geração dos pontos válidos
# Remove região sólida do degrau:
# x < 0 e y < 0
# -----------------------------
pts = []

for y in ys:
    for x in xs:
        if x < 0.0 and y < 0.0:
            continue

        pts.append((x, y, z))

pts = np.array(pts, dtype=float)

# -----------------------------
# Salvar no formato OpenFOAM vectorList
# -----------------------------
out_path = Path("gridPoints.xyz")

with out_path.open("w") as f:
    f.write(f"{len(pts)}\n")
    f.write("(\n")

    for x, y, z in pts:
        f.write(f"({x:.8f} {y:.8f} {z:.8f})\n")

    f.write(")\n")

print(f"Gerado {out_path} com {len(pts)} pontos válidos.")
print(f"Grade original: {nx} x {ny} = {nx * ny} pontos.")
print(f"Pontos removidos: {nx * ny - len(pts)}.")

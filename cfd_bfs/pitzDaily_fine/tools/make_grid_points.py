import numpy as np

# limites (m)
xmin, xmax = -0.0206, 0.290
ymin, ymax = -0.0254, 0.0254
z = 0.0

# resolução da grade (ajuste aqui)
nx, ny = 400, 160   # começa com isso; se ficar pesado, reduza

xs = np.linspace(xmin, xmax, nx)
ys = np.linspace(ymin, ymax, ny)

pts = np.array([(x, y, z) for y in ys for x in xs], dtype=float)

# formato de "vector list" usado pelo sample
with open("gridPoints.xyz", "w") as f:
    f.write(f"{len(pts)}\n(\n")
    for x, y, z in pts:
        f.write(f"({x:.8f} {y:.8f} {z:.8f})\n")
    f.write(")\n")

print(f"Gerado gridPoints.xyz com {len(pts)} pontos ({nx}x{ny}).")

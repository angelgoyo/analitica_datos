"""Diagrama de las seis fases del proyecto (CRISP-DM) y su relación con los objetivos específicos.

Es un diagrama conceptual: no depende de los datos. Se genera con código para que
ninguna figura del informe se dibuje a mano. El estado de cada fase se pasa como
argumento y se actualiza en cada entrega.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AZUL, NARANJA, GRIS, TINTA = "#2a78d6", "#eb6834", "#8f8f8a", "#222222"

FASES = [
    ("1. Comprensión\ndel negocio", "Objetivo 1", True),
    ("2. Comprensión\nde los datos", "Objetivo 1", True),
    ("3. Preparación\nde los datos", "Objetivo 1", True),
    ("4. Modelado", "Objetivo 2", False),
    ("5. Evaluación", "Objetivo 2", False),
    ("6. Despliegue\n(prototipo)", "Objetivos 3 y 4", False),
]


def dibujar(ruta, fases=FASES):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.12, 1.45)
    ax.axis("off")
    ancho, alto = 0.86, 0.62
    for i, (nombre, objetivo, hecha) in enumerate(fases):
        x = i + 0.07
        fc, ec = ("#eaf2fc", AZUL) if hecha else ("#f4f4f2", GRIS)
        ax.add_patch(FancyBboxPatch((x, 0.72), ancho, alto, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.1))
        ax.text(x + ancho / 2, 1.15, nombre, ha="center", va="center", fontsize=7, fontweight="bold", color=TINTA)
        ax.text(x + ancho / 2, 0.86, objetivo, ha="center", va="center", fontsize=6.3, color=TINTA)
        ax.text(x + ancho / 2, 0.57, "Completada" if hecha else "Pendiente", ha="center", va="center",
                fontsize=6.3, fontweight="bold", color=AZUL if hecha else GRIS)
        if i < len(fases) - 1:
            ax.add_patch(FancyArrowPatch((x + ancho + 0.01, 1.03), (x + 1 - 0.01, 1.03), arrowstyle="-|>", mutation_scale=8, color=GRIS, lw=1))
    # arco de retroalimentación típico de CRISP-DM entre evaluación y comprensión del negocio
    ax.add_patch(FancyArrowPatch((4.5, 0.40), (0.5, 0.40), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>", mutation_scale=8, color=NARANJA, lw=1, linestyle=(0, (4, 2))))
    ax.text(2.5, -0.02, "Si la evaluación no responde la pregunta, se vuelve a revisar el problema", ha="center", va="center", fontsize=6.3, color=NARANJA)
    ax.set_title("Fases del proyecto y objetivos específicos que cubre cada una", fontsize=9, fontweight="bold", pad=2)
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


if __name__ == "__main__":
    import sys
    dibujar(sys.argv[1] if len(sys.argv) > 1 else "fig_8_0_fases_proyecto.png")

"""Run the Dirichlet (PUBD) solver."""

from __future__ import annotations

from dirichlet_solver import dirichlet_solver


def main() -> None:
    _ = dirichlet_solver("examples2states2actions")


if __name__ == "__main__":
    main()

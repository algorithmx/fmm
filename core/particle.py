"""
Particle Module

Represents a single particle in the FMM with position, charge, and potential.
"""

from typing import Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Particle:
    """
    Represents a particle in the N-body simulation.

    Attributes:
        position: Particle coordinates (x, y) for 2D or (x, y, z) for 3D
        charge: Source strength/charge of the particle
        potential: Computed potential at this particle (initially None)
        index: Unique identifier for the particle
    """
    position: np.ndarray
    charge: float
    potential: Optional[float] = None
    index: int = 0

    def __post_init__(self):
        """Validate particle properties after initialization."""
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.potential is None:
            self.potential = 0.0

    @property
    def dim(self) -> int:
        """Return the spatial dimension of the particle."""
        return len(self.position)

    def distance_to(self, other: 'Particle') -> float:
        """Compute Euclidean distance to another particle."""
        return np.linalg.norm(self.position - other.position)

    def reset_potential(self):
        """Reset potential to zero."""
        self.potential = 0.0

    def __repr__(self) -> str:
        return f"Particle(id={self.index}, pos={self.position}, q={self.charge:.3f})"

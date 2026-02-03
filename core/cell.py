"""
Cell Module

Represents a cell in the hierarchical tree structure for FMM.
"""

from typing import List, Optional, Set
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class CellType(Enum):
    """Type of cell in the tree."""
    ROOT = "root"
    INTERNAL = "internal"
    LEAF = "leaf"


@dataclass
class Cell:
    """
    Represents a cell in the hierarchical tree decomposition.

    A cell represents a region in space containing particles.
    It forms the nodes of the quadtree/octree structure.

    Attributes:
        center: Center coordinates of the cell
        size: Side length of the cell (assuming cubic/square cells)
        level: Tree level (0 = root)
        index: Unique identifier within level
        parent: Parent cell reference (None for root)
        children: List of child cells
        particles: List of particles contained in this cell (only for leaves)
        cell_type: Type of cell (root, internal, or leaf)
        dimension: Spatial dimension (2 or 3)
        cubature_order: Cubature order for high-frequency FMM (optional)
        direction_set: Direction set for directional FMM (optional)
    """
    center: np.ndarray
    size: float
    level: int
    index: int
    parent: Optional['Cell'] = None
    children: List['Cell'] = field(default_factory=list)
    particles: List['Particle'] = field(default_factory=list)
    cell_type: CellType = CellType.LEAF
    dimension: int = 2
    cubature_order: Optional[int] = None          # For high-frequency FMM
    direction_set: Optional[List[np.ndarray]] = None  # For directional FMM

    def __post_init__(self):
        """Validate and initialize cell properties."""
        self.center = np.asarray(self.center, dtype=np.float64)
        self._morton_index: Optional[int] = None

    @property
    def morton_index(self) -> int:
        """Get Morton (Z-order) index for space-filling curve ordering."""
        if self._morton_index is None:
            self._morton_index = self._compute_morton_index()
        return self._morton_index

    def _compute_morton_index(self) -> int:
        """Compute Morton index by interleaving bits of spatial coordinates."""
        # For simplicity, we use a basic implementation
        # In production, use proper bit interleaving
        return self.index

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the minimum and maximum coordinates of the cell."""
        half_size = self.size / 2.0
        return self.center - half_size, self.center + half_size

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf cell."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Check if this is the root cell."""
        return self.parent is None

    @property
    def num_particles(self) -> int:
        """Return the number of particles in this cell."""
        if self.is_leaf:
            return len(self.particles)
        return sum(child.num_particles for child in self.children)

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside this cell."""
        min_bound, max_bound = self.bounds
        return np.all((point >= min_bound) & (point <= max_bound))

    def distance_to(self, other: 'Cell') -> float:
        """
        Compute the minimum distance between two cells.
        Returns 0 if cells overlap or touch.
        """
        min1, max1 = self.bounds
        min2, max2 = other.bounds

        # Calculate distance between bounds
        diff = np.zeros_like(min1)
        for i in range(len(min1)):
            if max1[i] < min2[i]:
                diff[i] = min2[i] - max1[i]
            elif max2[i] < min1[i]:
                diff[i] = min1[i] - max2[i]
            else:
                diff[i] = 0.0

        return np.linalg.norm(diff)

    def is_adjacent(self, other: 'Cell') -> bool:
        """
        Check if two cells are adjacent (touch or overlap).
        Used for determining near-field interactions.
        """
        return self.distance_to(other) == 0.0

    def is_well_separated(self, other: 'Cell', theta: float = 1.0) -> bool:
        """
        Check if two cells are well-separated using the Multipole Acceptance Criterion.

        MAC: r / max(size1, size2) >= theta
        where r is the distance between cell centers and theta is the opening angle.

        Args:
            other: The other cell to check separation with
            theta: Opening angle parameter (default 1.0)

        Returns:
            True if cells are well-separated, False otherwise
        """
        r = np.linalg.norm(self.center - other.center)
        max_size = max(self.size, other.size)
        return (r / max_size) >= theta

    def get_neighbors(self, max_depth: Optional[int] = None) -> Set['Cell']:
        """
        Get all adjacent cells at the same or nearby levels.

        Args:
            max_depth: Maximum level difference to consider (default: same level only)

        Returns:
            Set of neighboring cells
        """
        neighbors = set()
        self._collect_neighbors(neighbors, self.level, max_depth)
        return neighbors

    def _collect_neighbors(self, neighbors: Set['Cell'], target_level: int,
                          max_depth: Optional[int]):
        """Recursively collect neighboring cells."""
        if self.is_leaf or self.level == target_level:
            # At target level, check adjacency
            if self.is_adjacent(self) and len(neighbors) > 0:
                neighbors.add(self)
        else:
            # Continue down the tree
            for child in self.children:
                child._collect_neighbors(neighbors, target_level, max_depth)

    def get_interaction_list(self) -> List['Cell']:
        """
        Get the interaction list: cells at the same level that are well-separated
        but whose parents are not well-separated.

        This is key for the FMM M2L operator.
        """
        if self.is_root:
            return []

        interaction_list = []
        parent = self.parent
        if parent is None:
            return interaction_list

        # Get parent's neighbors
        parent_neighbors = parent.get_neighbors()

        # For each parent neighbor, get its children
        for parent_neighbor in parent_neighbors:
            for child in parent_neighbor.children:
                if (child is not self and
                    not self.is_adjacent(child) and
                    self.is_well_separated(child)):
                    interaction_list.append(child)

        return interaction_list

    def subdivide(self, num_children: int = 4) -> List['Cell']:
        """
        Subdivide this cell into child cells.

        Args:
            num_children: Number of children (4 for quadtree, 8 for octree)

        Returns:
            List of newly created child cells
        """
        if self.dimension == 2:
            return self._subdivide_2d()
        else:
            return self._subdivide_3d()

    def _subdivide_2d(self) -> List['Cell']:
        """Subdivide into 4 quadrants (quadtree)."""
        if self.dimension != 2:
            raise ValueError("Cannot use 2D subdivision for non-2D cell")

        half_size = self.size / 2.0
        quarter_size = half_size / 2.0

        # Define offsets for 4 quadrants
        offsets = [
            (-quarter_size, -quarter_size),  # Bottom-left
            (quarter_size, -quarter_size),   # Bottom-right
            (-quarter_size, quarter_size),   # Top-left
            (quarter_size, quarter_size),    # Top-right
        ]

        self.children = []
        for i, (dx, dy) in enumerate(offsets):
            new_center = self.center + np.array([dx, dy])
            child = Cell(
                center=new_center,
                size=half_size,
                level=self.level + 1,
                index=i,
                parent=self,
                dimension=self.dimension,
                cell_type=CellType.LEAF
            )
            self.children.append(child)

        self.cell_type = CellType.INTERNAL
        return self.children

    def _subdivide_3d(self) -> List['Cell']:
        """Subdivide into 8 octants (octree)."""
        if self.dimension != 3:
            raise ValueError("Cannot use 3D subdivision for non-3D cell")

        half_size = self.size / 2.0
        quarter_size = half_size / 2.0

        # Define offsets for 8 octants
        offsets = [
            (-quarter_size, -quarter_size, -quarter_size),
            (quarter_size, -quarter_size, -quarter_size),
            (-quarter_size, quarter_size, -quarter_size),
            (quarter_size, quarter_size, -quarter_size),
            (-quarter_size, -quarter_size, quarter_size),
            (quarter_size, -quarter_size, quarter_size),
            (-quarter_size, quarter_size, quarter_size),
            (quarter_size, quarter_size, quarter_size),
        ]

        self.children = []
        for i, (dx, dy, dz) in enumerate(offsets):
            new_center = self.center + np.array([dx, dy, dz])
            child = Cell(
                center=new_center,
                size=half_size,
                level=self.level + 1,
                index=i,
                parent=self,
                dimension=self.dimension,
                cell_type=CellType.LEAF
            )
            self.children.append(child)

        self.cell_type = CellType.INTERNAL
        return self.children

    def __repr__(self) -> str:
        type_str = self.cell_type.value
        return (f"Cell({type_str}, level={self.level}, idx={self.index}, "
                f"center={self.center}, size={self.size:.3f}, n={self.num_particles})")

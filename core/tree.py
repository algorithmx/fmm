"""
Tree Module

Implements hierarchical tree decomposition (quadtree/octree) for FMM.
"""

from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .cell import Cell, CellType
from .particle import Particle


@dataclass
class TreeConfig:
    """Configuration for tree construction."""
    max_depth: int = 10          # Maximum tree depth
    ncrit: int = 50              # Maximum particles per leaf (adaptive refinement)
    dimension: int = 2           # Spatial dimension (2 or 3)
    expansion_order: int = 4     # Order of multipole/local expansions
    theta: float = 1.0           # MAC opening angle parameter

    # High-frequency FMM enhancements
    use_high_frequency_m2l: bool = False           # Use diagonal M2L for high-frequency Helmholtz
    use_spherical_interpolation: bool = False      # Use spherical interpolation for M2M/L2L
    interpolation_method: str = 'direct'           # 'direct' or 'fft'
    adaptive_order: bool = False                   # Adapt expansion order with frequency
    use_fft_m2l: bool = True                       # Use FFT-based M2L for kernel-independent
    use_directional: bool = False                  # Use directional FMM for extreme high-frequency
    directional_threshold_kw: float = 10.0         # Threshold κw for activating directional FMM
    min_order: int = 2                             # Minimum expansion order for adaptive
    max_order: int = 30                            # Maximum expansion order for adaptive
    accuracy_digits: float = 6.0                   # d₀: requested digits of accuracy (Chapter 3)

    def __post_init__(self):
        """Validate configuration."""
        if self.dimension not in [2, 3]:
            raise ValueError("Dimension must be 2 or 3")
        if self.max_depth <= 0:
            raise ValueError("Max depth must be positive")
        if self.ncrit <= 0:
            raise ValueError("Ncrit must be positive")
        if self.expansion_order <= 0:
            raise ValueError("Expansion order must be positive")
        if self.interpolation_method not in ['direct', 'fft']:
            raise ValueError("Interpolation method must be 'direct' or 'fft'")
        if self.min_order < 0:
            raise ValueError("Min order must be non-negative")
        if self.max_order < self.min_order:
            raise ValueError("Max order must be >= min order")
        if self.directional_threshold_kw < 0:
            raise ValueError("Directional threshold must be non-negative")


class Tree:
    """
    Hierarchical tree structure for FMM decomposition.

    Supports both uniform (max-depth based) and adaptive (ncrit-based)
    refinement strategies.
    """

    def __init__(self, particles: List[Particle], config: TreeConfig):
        """
        Initialize the tree with particles.

        Args:
            particles: List of particles to organize in the tree
            config: Tree configuration parameters
        """
        self.particles = particles
        self.config = config
        self.root: Optional[Cell] = None
        self.leaves: List[Cell] = []
        self.cells_by_level: List[List[Cell]] = []

        if particles:
            self._build_tree()

    def _compute_bounding_box(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the bounding box for all particles.

        Returns:
            Tuple of (center, extents, size) where:
            - center: Center of the bounding box
            - extents: Half-width in each dimension
            - size: Maximum side length
        """
        positions = np.array([p.position for p in self.particles])

        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)

        center = (min_coords + max_coords) / 2.0
        extents = (max_coords - min_coords) / 2.0

        # Use maximum extent to ensure square/cubic cells
        size = 2.0 * np.max(extents)

        # Add small padding to avoid boundary issues
        size *= 1.01

        return center, extents, size

    def _build_tree(self):
        """Build the hierarchical tree structure."""
        # Compute initial bounding box
        center, _, size = self._compute_bounding_box()

        # Create root cell
        self.root = Cell(
            center=center,
            size=size,
            level=0,
            index=0,
            dimension=self.config.dimension,
            cell_type=CellType.ROOT
        )

        # Insert all particles into the tree
        for particle in self.particles:
            self._insert_particle(self.root, particle)

        # Build cells_by_level index
        self._index_levels()

        # Collect leaves
        self._collect_leaves()

    def _insert_particle(self, cell: Cell, particle: Particle):
        """
        Recursively insert a particle into the tree.

        Uses adaptive subdivision based on ncrit parameter.
        """
        if cell.is_leaf:
            cell.particles.append(particle)

            # Check if we need to subdivide
            should_subdivide = (
                len(cell.particles) > self.config.ncrit and
                cell.level < self.config.max_depth
            )

            if should_subdivide:
                self._subdivide_cell(cell)
        else:
            # Find the appropriate child
            for child in cell.children:
                if child.contains(particle.position):
                    self._insert_particle(child, particle)
                    break

    def _subdivide_cell(self, cell: Cell):
        """
        Subdivide a leaf cell and redistribute its particles.
        """
        # Create children
        cell.subdivide()

        # Redistribute particles to children
        for particle in cell.particles:
            for child in cell.children:
                if child.contains(particle.position):
                    child.particles.append(particle)
                    break

        # Clear particles from parent (now internal node)
        cell.particles = []

        # Recursively check if children need further subdivision
        for child in cell.children:
            if len(child.particles) > self.config.ncrit and child.level < self.config.max_depth:
                self._subdivide_cell(child)

    def _index_levels(self):
        """Build an index of cells by their level for efficient traversal."""
        max_level = 0
        cell_count = {0: 1}

        # BFS to count levels
        from collections import deque
        queue = deque([self.root])

        while queue:
            cell = queue.popleft()
            max_level = max(max_level, cell.level)

            if not cell.is_leaf:
                for child in cell.children:
                    queue.append(child)

        # Initialize level index
        self.cells_by_level = [[] for _ in range(max_level + 1)]

        # Populate level index
        queue = deque([self.root])
        while queue:
            cell = queue.popleft()
            self.cells_by_level[cell.level].append(cell)

            if not cell.is_leaf:
                for child in cell.children:
                    queue.append(child)

    def _collect_leaves(self):
        """Collect all leaf cells."""
        self.leaves = []
        self._traverse_collect_leaves(self.root)

    def _traverse_collect_leaves(self, cell: Cell):
        """Recursively collect leaf cells."""
        if cell.is_leaf:
            self.leaves.append(cell)
        else:
            for child in cell.children:
                self._traverse_collect_leaves(child)

    def get_max_level(self) -> int:
        """Return the maximum tree level."""
        return len(self.cells_by_level) - 1

    def get_cells_at_level(self, level: int) -> List[Cell]:
        """Get all cells at a specific level."""
        if 0 <= level < len(self.cells_by_level):
            return self.cells_by_level[level]
        return []

    def get_near_field_neighbors(self, cell: Cell) -> List[Cell]:
        """
        Get the near-field neighbors for a cell.

        Near-field neighbors are adjacent cells at the same level
        that require direct (P2P) computation.
        """
        if cell.is_root:
            return []

        neighbors = []
        same_level_cells = self.get_cells_at_level(cell.level)

        for other in same_level_cells:
            if other is not cell and other.is_adjacent(cell):
                neighbors.append(other)

        return neighbors

    def get_interaction_list(self, cell: Cell) -> List[Cell]:
        """
        Get the interaction list for FMM M2L operator.

        Returns cells at the same level that are well-separated
        but whose parents were not well-separated.
        """
        if cell.level == 0:
            return []

        interaction_list = []
        parent = cell.parent
        if parent is None:
            return interaction_list

        # Get parent's neighbors
        parent_level = cell.level - 1
        parent_neighbors = self.get_cells_at_level(parent_level)

        for parent_neighbor in parent_neighbors:
            if parent_neighbor is parent:
                continue

            # Check if parents are well-separated
            if parent.is_well_separated(parent_neighbor, self.config.theta):
                continue

            # Children of this parent neighbor are in interaction list
            if parent_neighbor.is_leaf:
                if (not cell.is_adjacent(parent_neighbor) and
                    cell.is_well_separated(parent_neighbor, self.config.theta)):
                    interaction_list.append(parent_neighbor)
            else:
                for child in parent_neighbor.children:
                    if (child is not cell and
                        not cell.is_adjacent(child) and
                        cell.is_well_separated(child, self.config.theta)):
                        interaction_list.append(child)

        return interaction_list

    def dual_tree_traversal(self, source_tree: 'Tree', target_cell: Cell,
                           source_cell: Cell, theta: float,
                           interaction_lists: dict, near_lists: dict) -> None:
        """
        Dual Tree Traversal (DTT) algorithm for adaptive FMM.

        From Chapter 2, Algorithm 6:
        DTT efficiently constructs interaction lists for adaptive trees by
        simultaneously traversing both source and target trees.

        Args:
            source_tree: The source tree
            target_cell: Current target cell being processed
            source_cell: Current source cell being processed
            theta: Multipole Acceptance Criterion parameter
            interaction_lists: Dictionary to store M2L interaction lists
            near_lists: Dictionary to store P2P near-field lists
        """
        # Check MAC criterion
        if target_cell.is_well_separated(source_cell, theta):
            # Well-separated: add to interaction list (M2L)
            if id(target_cell) not in interaction_lists:
                interaction_lists[id(target_cell)] = []
            interaction_lists[id(target_cell)].append(source_cell)
            return

        # Not well-separated
        if target_cell.is_leaf and source_cell.is_leaf:
            # Both leaves: add to near-field list (P2P)
            if id(target_cell) not in near_lists:
                near_lists[id(target_cell)] = []
            near_lists[id(target_cell)].append(source_cell)
        elif target_cell.is_leaf:
            # Target is leaf, source is not: recurse on source children
            for source_child in source_cell.children:
                self.dual_tree_traversal(
                    source_tree, target_cell, source_child,
                    theta, interaction_lists, near_lists
                )
        elif source_cell.is_leaf:
            # Source is leaf, target is not: recurse on target children
            for target_child in target_cell.children:
                self.dual_tree_traversal(
                    source_tree, target_child, source_cell,
                    theta, interaction_lists, near_lists
                )
        else:
            # Neither is leaf: recurse on both
            # For efficiency, we can use different strategies:
            # 1. Recurse on larger cell
            # 2. Recurse on both (full tree traversal)
            for target_child in target_cell.children:
                for source_child in source_cell.children:
                    self.dual_tree_traversal(
                        source_tree, target_child, source_child,
                        theta, interaction_lists, near_lists
                    )

    def construct_lists_dtt(self, other_tree: 'Tree' = None) -> tuple:
        """
        Construct interaction and near-field lists using Dual Tree Traversal.

        From Chapter 2, Algorithm 6:
        DTT provides an adaptive way to build all necessary lists for FMM.

        Args:
            other_tree: Optional second tree for different source/target distributions
                       If None, uses the same tree for both

        Returns:
            Tuple of (interaction_lists, near_lists) dictionaries
        """
        if other_tree is None:
            other_tree = self

        interaction_lists = {}
        near_lists = {}

        # Start traversal from roots
        self.dual_tree_traversal(
            other_tree,
            self.root,
            other_tree.root,
            self.config.theta,
            interaction_lists,
            near_lists
        )

        return interaction_lists, near_lists

    def get_uvwxy_lists(self, cell: Cell) -> dict:
        """
        Construct UVWXY-lists for adaptive FMM.

        From Chapter 2, Section 2.2.4.2:
        For adaptive trees, we need five types of lists:
        - U-list: Well-separated cells (M2L)
        - V-list: Parent's well-separated cells (M2L via parent)
        - W-list: Adjacent cells at same level (P2P)
        - X-list: Children of adjacent cells (P2P)
        - Y-list: Well-separated children of adjacent cells (M2L)

        Args:
            cell: The target cell to build lists for

        Returns:
            Dictionary with keys 'U', 'V', 'W', 'X', 'Y' containing respective lists
        """
        lists = {
            'U': [],  # M2L interaction list
            'V': [],  # Parent's M2L list
            'W': [],  # P2P adjacent cells at same level
            'X': [],  # P2P children of adjacent cells
            'Y': []   # M2L well-separated children
        }

        if cell.is_root:
            return lists

        parent = cell.parent
        if parent is None:
            return lists

        # Get parent's adjacent cells
        parent_level_cells = self.get_cells_at_level(parent.level)
        parent_adjacent = [c for c in parent_level_cells
                          if c is not parent and c.is_adjacent(parent)]

        # W-list: Adjacent cells at same level
        same_level_cells = self.get_cells_at_level(cell.level)
        for other in same_level_cells:
            if other is not cell and other.is_adjacent(cell):
                lists['W'].append(other)

        # U-list: Well-separated cells at same level
        for other in same_level_cells:
            if other is not cell and not other.is_adjacent(cell):
                if cell.is_well_separated(other, self.config.theta):
                    lists['U'].append(other)

        # V-list: Parent's well-separated cells
        for parent_neighbor in parent_adjacent:
            if parent.is_well_separated(parent_neighbor, self.config.theta):
                lists['V'].append(parent_neighbor)

        # X-list: Children of adjacent cells
        for parent_neighbor in parent_adjacent:
            if not parent_neighbor.is_leaf:
                for child in parent_neighbor.children:
                    if child.is_adjacent(cell) and child.level == cell.level:
                        lists['X'].append(child)

        # Y-list: Well-separated children of adjacent parents
        for parent_neighbor in parent_adjacent:
            if not parent_neighbor.is_leaf:
                for child in parent_neighbor.children:
                    if (not child.is_adjacent(cell) and
                        child.is_well_separated(cell, self.config.theta)):
                        lists['Y'].append(child)

        return lists

    def get_leaf_containing(self, position: np.ndarray) -> Optional[Cell]:
        """Find the leaf cell containing a given position."""
        return self._find_leaf(self.root, position)

    def _find_leaf(self, cell: Cell, position: np.ndarray) -> Optional[Cell]:
        """Recursively find the leaf containing a position."""
        if cell.is_leaf:
            return cell if cell.contains(position) else None

        for child in cell.children:
            if child.contains(position):
                return self._find_leaf(child, position)

        return None

    def get_statistics(self) -> dict:
        """
        Compute and return tree statistics.

        Returns:
            Dictionary with tree statistics
        """
        num_cells = sum(len(level) for level in self.cells_by_level)
        avg_particles_per_leaf = np.mean([leaf.num_particles for leaf in self.leaves])
        max_particles_per_leaf = max([leaf.num_particles for leaf in self.leaves])

        return {
            'num_particles': len(self.particles),
            'num_cells': num_cells,
            'num_leaves': len(self.leaves),
            'max_depth': self.get_max_level(),
            'avg_particles_per_leaf': avg_particles_per_leaf,
            'max_particles_per_leaf': max_particles_per_leaf,
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"Tree(dim={self.config.dimension}, "
                f"N={stats['num_particles']}, "
                f"cells={stats['num_cells']}, "
                f"leaves={stats['num_leaves']}, "
                f"depth={stats['max_depth']})")

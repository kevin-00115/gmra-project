# SYSTEM IMPORTS
from collections import Counter
from tqdm import tqdm
from typing import List, Set
import numpy as np


# PYTHON PROJECT IMPORTS
from .covertree import CoverTree, CoverTreeNode


DyadicCellType = "DyadicCell"


class DyadicCell(object):
    def __init__(self,
                 node: CoverTreeNode,
                 scale: int,
                 dataset_idx: int) -> None:
        self.parent: DyadicCellType = None
        self.children: List[DyadicCellType] = list()
        self.dataset_idx: int = dataset_idx
        self.scale: int = scale

        self.pt_idxs: np.ndarray = self.get_idxs(node, scale)

    def get_idxs(self,
                 node: CoverTreeNode,
                 max_scale: int) -> np.ndarray:
        C: List[CoverTreeNode] = [node]
        idxs: Set[int] = {node.pt_idx}
        for scale in range(max_scale, -1, -1):
            C_next: List[CoverTreeNode] = sum([c.get_children(scale) for c in C], [])
            idxs.update([c.pt_idx for c in C_next])
            C = C_next
        return np.array(list(idxs), dtype=int)

    def add_child(self,
                  child: DyadicCellType) -> DyadicCellType:
        self.children.append(child)
        child.parent = self

        return self


def unique(c) -> bool:
    return Counter(c).get(True, 0) == 1


class DyadicCellTree(object):
    def __init__(self) -> None:
        self.root: DyadicCell = None
        self.num_nodes: int = 0
        self.datasets: List[np.ndarray] = list()
        self.base: float = None
        self.num_levels: int = None

    def from_covertree(self,
                       tree: CoverTree) -> "DyadicCellTree":
        self.datasets = tree.datasets
        self.base = tree.base
        self.num_levels = tree.max_scale - tree.min_scale + 1

        layer_nodes: List[CoverTreeNode] = [tree.root]
        layer_cells: List[DyadicCell] = [DyadicCell(tree.root, tree.max_scale,
                                                    tree.root.dataset_idx)]

        self.root = layer_cells[0]

        for scale in reversed(range(tree.min_scale, tree.max_scale+1)):
            next_layer_nodes: List[CoverTreeNode] = list()
            next_layer_cells: List[DyadicCell] = list()

            for parent_node, parent_cell in zip(layer_nodes, layer_cells):
                for child_node in parent_node.get_children(scale):
                    next_layer_nodes.append(child_node)

                    child_cell: DyadicCell = DyadicCell(child_node, scale-1,
                                                        child_node.dataset_idx)
                    parent_cell.add_child(child_cell)
                    next_layer_cells.append(child_cell)
                    self.num_nodes += 1

            print("number of children at scale %s: %s" % (scale, len(next_layer_nodes)))

            layer_nodes = next_layer_nodes
            layer_cells = next_layer_cells
        return self


    def check_covering_invariant(self,
                                 parent_cell: DyadicCell,
                                 child_cell: DyadicCell,
                                 scale: int) -> bool:

        covering_check: bool = True
        for parent_idx in parent_cell.pt_idxs:
            for child_idx in child_cell.pt_idxs:
                covering_check = covering_check and \
                    (((self.datasets[parent_cell.dataset_idx][parent_idx] -
                       self.datasets[child_cell.dataset_idx][child_idx]) ** 2)
                    .sum()**(1/2) <= self.base**scale)

        return covering_check


    def check_tree(self) -> None:
        layer_cells: List[DyadicCell] = [self.root]

        for level in range(self.num_levels):
            print("num pts at level %s: " % level, np.unique(np.hstack([c.pt_idxs
                for c in layer_cells])).shape[0])

            next_layer_cells: List[DyadicCell] = sum([c.children for c in layer_cells], [])

            """
            if len(next_layer_cells) > 0:
                # only check invariants while children exist

                # check nesting
                idxs_in_current_layer = np.unique(np.hstack([c.pt_idxs
                                                             for c in layer_cells]))
                idxs_in_next_layer = np.unique(np.hstack([c.pt_idxs
                                                          for c in next_layer_cells]))
                nesting: bool = np.all(np.isin(idxs_in_current_layer, idxs_in_next_layer))
                                # idxs_in_next_layer <= idxs_in_current_layer

                # check covering
                covering: bool = all(self.check_covering_invariant(q, p, q.scale)
                                     for q in layer_cells for p in next_layer_cells)

                # if not nesting:
                #     raise ValueError("ERROR: nesting fails at scale %s" % q.scale)
                # if not covering:
                #     raise ValueError("ERROR: covering fails at scale %s" % q.scale)
            """

            layer_cells = next_layer_cells


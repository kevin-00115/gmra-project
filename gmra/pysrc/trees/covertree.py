# SYSTEM IMPORTS
from tqdm import tqdm
from itertools import product
from collections import Counter
from heapq import nsmallest
from random import choice
from typing import Dict, List, Set, Tuple
import numpy as np
import pickle as pk

# PYTHON PROJECT IMPORTS


CoverTreeNodeType = "CoverTreeNode"


class CoverTreeNode(object):
    def __init__(self,
                 pt_idx: int = None,
                 dataset_idx: int = None) -> None:
        self.pt_idx: int = pt_idx
        self.dataset_idx: int = dataset_idx
        self.parent: CoverTreeNodeType = None
        self.children: Dict[int, List[CoverTreeNodeType]] = dict()

    def add_child(self,
                  child: CoverTreeNodeType,
                  scale: int) -> CoverTreeNodeType:
        if scale not in self.children:
            self.children[scale] = list()
        self.children[scale].append(child)
        child.parent = self

        return self

    def get_children(self,
                     scale: int,
                     only_children: bool = False) -> Set[CoverTreeNodeType]:
        all_children: Set[CoverTreeNodeType] = list()
        if not only_children:
            all_children.append(self)

        all_children.extend(self.children.get(scale, list()))

        return all_children

    def disconnect_from_parent(self,
                               scale: int) -> CoverTreeNodeType:
        if self.parent is not None:
            self.parent.children[scale+1].remove(self)
            self.parent = None

        return self


def unique(c) -> bool:
    return Counter(c).get(True, 0) == 1


class CoverTree(object):
    def __init__(self,
                 max_scale: int = 10,
                 base: float = 2) -> None:

        self.max_scale: int = int(max_scale)
        self.min_scale: int = self.max_scale
        if self.max_scale < 0:
            raise ValueError("ERROR: max_scale must be >= 0")

        self.num_pts: int = 0
        self.root: CoverTreeNode = None
        self.base: float = float(base)

        self.datasets: List[np.ndarray] = list()
        self.node_registry: Set[CoverTreeNode] = set()

    @property
    def size(self) -> int:
        return self.num_pts

    def insert(self,
               dataset: np.ndarray) -> "CoverTree":
        self.datasets.append(dataset)
        for pt_idx in tqdm(range(dataset.shape[0])):
            self._insert(pt_idx, len(self.datasets)-1)
        return self

    def _insert(self,
                pt_idx: int,
                dataset_idx: int) -> None:
        if self.root is None:
            self.root = self._make_node(pt_idx, dataset_idx)
        else:
            self._insert_nonroot(pt_idx, dataset_idx)

    def _make_node(self,
                   pt_idx: int,
                   dataset_idx: int) -> CoverTreeNode:
        node: CoverTreeNode = CoverTreeNode(pt_idx, dataset_idx)
        self.node_registry.add(node)
        self.num_pts += 1
        return node

    def _load_pt(self,
                 n: CoverTreeNode) -> np.ndarray:
        return self.datasets[n.dataset_idx][n.pt_idx]

    def _insert_nonroot(self,
                        pt_idx: int,
                        dataset_idx: int) -> None:
        pt: np.ndarray = self.datasets[dataset_idx][pt_idx]

        Qi_p_ds: List[Tuple[CoverTreeNode, float]] = [(self.root,
                                                       self._dist_func(pt,
                                                        self._load_pt(self.root)),)]
        scale: int = self.max_scale
        p_scale: float = None
        parent: CoverTreeNode = None

        stop: bool = False
        while not stop:
            Q_p_ds = self._get_children_distribution(pt, Qi_p_ds, scale)
            d_p_Q: float = self._min_ds(Q_p_ds)

            if d_p_Q == 0:
                return
            elif d_p_Q > self.base**scale:
                stop = True
            else:

                if self._min_ds(Qi_p_ds) <= self.base**scale:
                    parent = choice([q for q,d in Qi_p_ds if d <= self.base**scale])
                    p_scale = scale

                Qi_p_ds: List[Tuple[CoverTreeNode, float]] = [(q,d) for q,d in Q_p_ds
                                                              if d <= self.base**scale]
                scale -= 1

        parent.add_child(self._make_node(pt_idx, dataset_idx), p_scale)
        self.min_scale = min(self.min_scale, p_scale-1)

    def _dist_func(self,
                   pt1: np.ndarray,
                   pt2: np.ndarray) -> float:
        # l-2 distance
        return np.sqrt(((pt1-pt2)**2).sum())

    def _get_children_distribution(self,
                                   pt: np.ndarray,
                                   Qi_p_ds: List[Tuple[CoverTreeNode, float]],
                                   scale: float) -> List[Tuple[CoverTreeNode, float]]:
        Q: List[Tuple[CoverTreeNode, float]] = sum([n.get_children(scale,
                                                                   only_children=True)
                                                    for n, _ in Qi_p_ds], [])
        Q_p_ds: List[Tuple[CoverTreeNode, float]] = [(q, self._dist_func(pt,
                                                            self._load_pt(q)),)
                                                     for q in Q]
        return Qi_p_ds + Q_p_ds

    def _kmin_p_ds(self,
                   k: int,
                   Q_p_ds: List[Tuple[CoverTreeNode, float]]) -> float:
        return nsmallest(k, Q_p_ds, lambda x: x[1])

    def _min_ds(self,
                Q_p_ds: List[Tuple[CoverTreeNode, float]]) -> float:
        return self._kmin_p_ds(1, Q_p_ds)[0][1]

    def check_tree(self) -> None:
        C: List[CoverTreeNode] = [self.root]
        for scale in reversed(range(self.min_scale, self.max_scale+1)):
            C_next: List[CoverTreeNode] = sum([p.get_children(scale) for p in C], [])

            # check invariants
            if not set(C) <= set(C_next):
                raise ValueError("ERROR at scale %s, nesting fails" % scale)
            if not all(unique(self._dist_func(self._load_pt(p), self._load_pt(q)) <=
                              self.base**scale and p in q.get_children(scale)
                              for q in C)
                       for p in C_next):
                raise ValueError("ERROR at scale %s, covering fails" % scale)
            if not all(self._dist_func(self._load_pt(p), self._load_pt(q)) >
                       self.base**scale
                       for p,q in product(C, C)
                       if p != q):
                raise ValueError("ERROR at scale %s, separation fails" % scale)

            C = C_next

    def save(self, fname):
      pk.dump(self, open(fname, "wb"))
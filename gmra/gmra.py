import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import argparse
import os
import sys
import time
import copy
from tqdm import tqdm

# Import GMRA modules
from pysrc.trees.wavelettree import WaveletTree
from pysrc.utils.inverse import *
from pysrc.utils.utils import *

from mcas_gmra import CoverTree, DyadicTree

def read_data(file_path):
    """Read data from a file and return it as a PyTorch tensor."""
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            float_values = [float(val) for val in values[1:]]
            data_list.append(float_values)
    return pt.tensor(data_list, dtype=pt.float32)


# ---------------------- GMRA Class ----------------------
class GMRA:
    def __init__(self, threshold=0.1):
        """
        Initialize the GMRA object.
        
        Parameters:
            threshold (float): Threshold parameter for the WaveletTree.
        """
        self.threshold = threshold

    def fit(self, cover_tree, X):
        """
        Build the GMRA tree from the data matrix X.
        This includes constructing the cover tree, dyadic tree, and wavelet tree.
        """
        self.X = X
        self.cover_tree = cover_tree
        self.dyadic_tree = DyadicTree(self.cover_tree)
        self.wavelet_tree = WaveletTree(
            self.dyadic_tree, self.X, 0, self.X.shape[-1],
            inverse=True, thresholds=self.threshold
        )
        self.wavelet_tree.make_wavelets(X)

    def fgwt(self):
        """
        Compute the forward GMRA transform (generalized forward wavelet transform)
        on the data X and return the multi-scale coefficients.
        """
        X = self.X
        leafs = get_leafs(self.wavelet_tree.root)
        Qjx = [None] * X.shape[0]
        for leaf in leafs:
            data_idx = int(leaf.idxs[0])
            # Convert leaf attributes to torch tensors.
            basis = pt.tensor(leaf.basis, dtype=X.dtype, device=X.device)
            center = pt.tensor(leaf.center, dtype=X.dtype, device=X.device)
            wav_basis = pt.tensor(leaf.wav_basis, dtype=X.dtype, device=X.device)
            
            # Compute local projection using the converted basis and center.
            pjx = basis @ (X[data_idx:data_idx+1, :].T - center)
            qjx = wav_basis @ basis.T @ pjx
            Qjx[data_idx] = [qjx]
            pJx = pjx
            p = path(leaf)
            # For each intermediate node in the path (except root and leaf)
            for n in reversed(p[1:-1]):
                # Convert intermediate node attributes to torch tensors.
                n_basis = pt.tensor(n.basis, dtype=X.dtype, device=X.device)
                n_center = pt.tensor(n.center, dtype=X.dtype, device=X.device)
                n_wav_basis = pt.tensor(n.wav_basis, dtype=X.dtype, device=X.device)
                
                pjx = n_basis @ basis.T @ pJx + n_basis @ (center - n_center)
                qjx = n_wav_basis @ n_basis.T @ pjx
                Qjx[data_idx].append(qjx)
            # Process the root node.
            n = p[0]
            n_basis = pt.tensor(n.basis, dtype=X.dtype, device=X.device)
            n_center = pt.tensor(n.center, dtype=X.dtype, device=X.device)
            pjx = n_basis @ basis.T @ pJx + n_basis @ (center - n_center)
            qjx = pjx  # At the root, synthesis yields the final coefficient.
            Qjx[data_idx].append(qjx)
            Qjx[data_idx] = list(reversed(Qjx[data_idx]))
        return Qjx

    def igwt(self, gmra_q_coeff):
        """
        Compute the inverse GMRA transform using the coefficients stored in gmra_q_coeff
        and reconstruct the data.
        """
        X = self.X
        wavelet_tree = self.wavelet_tree
        X_recon = np.zeros_like(X)
        for leaf in get_leafs(wavelet_tree.root):
            data_idx = leaf.idxs[0]
            chain = path(leaf)
            ct = -1
            Qjx = leaf.wav_basis.T @ gmra_q_coeff[data_idx][ct] + leaf.wav_consts
            new_chain = chain[1:-1]
            for jj, n in reversed(list(enumerate(new_chain))):
                ct -= 1
                Qjx += (n.wav_basis.T @ gmra_q_coeff[data_idx][ct] +
                        n.wav_consts +
                        new_chain[jj-1].basis.T @ new_chain[jj-1].basis @ Qjx)
            ct -= 1
            Qjx += chain[0].basis.T @ gmra_q_coeff[data_idx][ct] + chain[0].center
            X_recon[data_idx:data_idx+1, :] = Qjx.T
        self.X_recon = X_recon
        return X_recon


def main():
    init_time = time.time()
    parser = argparse.ArgumentParser(description="GMRA Embedding Extraction using fgwt")
    # Note: We no longer require a cover tree file because GMRA.fit builds it from data.
    parser.add_argument("--covertree_path", type=str, 
                        default= "/scratch/kevinyang/gmra-project/gmra/example/SNAP_facebook/facebook_combined_dim_256.json",
                        help="Path to serialized cover tree JSON file")
    parser.add_argument("--data_file", type=str,
                        default="/scratch/kevinyang/gmra-project/gmra/example/SNAP_facebook/facebook_combined_dim_256.csv",
                        help="Path to the data file")
    
    parser.add_argument("--scale_index", type=int, default=0, help="Scale index for embedding extraction")
    args = parser.parse_args()

    # Load data and node IDs.
    print("Loading data...")
    start_time = time.time()
    node_ids = []
    X_data = []
    with open(args.data_file, 'r') as file:
        next(file)  # Skip metadata line
        for line in file:
            parts = line.strip().split()
            node_ids.append(parts[0])
            X_data.append(list(map(float, parts[1:])))
    X = pt.tensor(X_data, dtype=pt.float32)
    print(f"Loaded {len(node_ids)} node IDs and data in {time.time()-start_time:.4f} seconds")
    print("First 5 node IDs:", node_ids[:5])

    # Build GMRA model from the data.
    print("Building GMRA model...")
    start_time = time.time()
    gmra_instance = GMRA(threshold=0.1)
    cover_tree = CoverTree(args.covertree_path)
    gmra_instance.fit(cover_tree, X)
    print(f"GMRA model built in {time.time()-start_time:.4f} seconds")
    
    
    # Compute the forward GMRA transform using the GMRA.fgwt method.
    print("Computing the forward GMRA transform (fgwt)...")
    start_time = time.time()
    gmra_q_coeff = gmra_instance.fgwt()
    print(f"Computed fgwt coefficients in {time.time()-start_time:.4f} seconds")

    # Extract low-dimensional embeddings at the specified scale.
    print(f"Extracting embeddings at scale index {args.scale_index}...")
    embeddings = []
    for coeffs in gmra_q_coeff:
        if args.scale_index < len(coeffs):
            embedding = coeffs[args.scale_index]
        else:
            embedding = coeffs[-1]
        embeddings.append(embedding.flatten().tolist())
    embeddings_with_ids = [[node_id] + emb for node_id, emb in zip(node_ids, embeddings)]

    # Save the embeddings to file.
    output_dir = "./reduced_embeddings/256"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reduced_dimension_fgwt.txt")
    print("Saving embeddings with node IDs...")
    with open(output_path, 'w') as file:
        file.write(f"{len(embeddings_with_ids)} {len(embeddings_with_ids[0]) - 1}\n")
        for embedding in embeddings_with_ids:
            file.write(" ".join(map(str, embedding)) + "\n")
    print(f"Low-dimensional embeddings saved to {output_path}")
    print(f"Total script runtime: {time.time()-init_time:.4f} seconds")


if __name__ == "__main__":
    main()
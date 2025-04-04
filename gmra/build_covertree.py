# SYSTEM IMPORTS
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt
import pickle as pk
import smtplib
import time  # Added to track time

from pysrc.trees.covertree import CoverTree

class CoverTreeBuilder:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
            
    def create_json(output_dir):
        # Extract the base filename from the provided path
        base_filename = os.path.basename(output_dir)

        # Remove the extension from the base filename
        root, _ = os.path.splitext(base_filename)

        # Create the new filename with the ".json" extension
        json_filename = f"{root}.json"

        return json_filename
    
    def read_data(file_path):
        # Initialize an empty list to store the data
        data_list = []

        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line into entries using space as the delimiter
                entries = line.strip().split(' ')
                
                # Skip the first entry (identifier) and convert subsequent entries to float
                float_values = [float(entry) for entry in entries[1:]]
                data_list.append(float_values)

        # Added to filter out any values that are not of the same length [Expected: 256]
        data_list = [data for data in data_list if len(data) == 2048]
        # Convert the list of lists to a PyTorch tensor
        tensor_data = pt.tensor(data_list)

        return tensor_data
    
    def build_cover_tree(self, output_dir):
        print("loading data")
        X_pt = self.read_data(self.data_dir)
        print("done")
        
        cover_tree = CoverTree(max_scale = 8)
        
        # start time tracking 
        start_time = time.time()
        print("building cover tree")
        for pt_idx in tqdm(range(X_pt.shape[0]), desc = "building covertree"):
            cover_tree.insert_pt(X_pt[pt_idx])
            
        # stop time tracking and calculate time taken 
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken to build cover tree: {time_taken:.2f} seconds")
        
        filename = self.create_json(output_dir)
        filepath = os.path.join(output_dir, filename)
        print("serializing covertree to [%s]" % filepath)
        cover_tree.save(filepath)
        # Send email after completing the process
        self.email(success=True, dimensions=X_pt.shape[1], data_dir=self.data_dir, time_taken=time_taken)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="directory to where covertree will serialize itself to")
    parser.add_argument("--data_file", type=str, 
                        help="path to the data file")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    builder = CoverTreeBuilder(args.data_file)
    builder.build_cover_tree(args.data_dir)
    
if __name__ == "__main__":
    main()
import os
import pickle

def Find_project_root(marker='.idea'):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    while not os.path.exists(os.path.join(current_dir, marker)):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"'{marker}' folder not found. Are you sure you're in the correct project?")
        current_dir = parent_dir

    return current_dir
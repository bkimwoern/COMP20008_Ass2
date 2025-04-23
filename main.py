"""
COMP20008 Elements of Data Processing
2025 Semester 1
Assignment 2

Solution: main file

DO NOT CHANGE THIS FILE!
"""

import os
import sys

def verify_preprocessing():
    try:
        from preprocessing import preprocessing
    except ImportError:
        print("Preprocessing's function not found.")
        return

    print("=" * 80)
    print("Executing preprocessing...\n")
    preprocessing()

    # print("Checking Task 1.1's output...\n")
    #print("Preprocessing has no expected output but executed without exceptions\n")

    print("Finished Preprocessing")
    print("=" * 80)


def main():
    args = sys.argv
    assert len(args) >= 2, "Please provide a task."
    task = args[1]
    valid_tasks = ["preprocessing"]
    assert task in valid_tasks, \
        f"Invalid task \"{task}\", options are: {valid_tasks}."
    if task == "preprocessing":
        verify_preprocessing()

if __name__ == "__main__":
    main()

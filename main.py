"""
COMP20008 Elements of Data Processing
2025 Semester 1
Assignment 1

Solution: main file

DO NOT CHANGE THIS FILE!
"""

import os
import sys


def verify_task1_1():
    try:
        from task1_1 import task1_1
    except ImportError:
        print("Task 1.1's function not found.")
        return

    print("=" * 80)
    print("Executing Task 1.1...\n")
    task1_1()

    # print("Checking Task 1.1's output...\n")
    print("Task 1.1 has no expected output but executed without exceptions\n")

    print("Finished Task 1.1")
    print("=" * 80)


def verify_task1_2():

    try:
        from task1_2 import task1_2
    except ImportError:
        print("Task 1.2's function not found.")
        return

    print("=" * 80)
    print("Executing Task 1.2...\n")
    task1_2()

    print("Checking Task 1.2's output...\n")

    for expected_file in ["task1_2_age.png", "task1_2_driver.png", "task1_2_seat.png"]:
        if os.path.isfile(expected_file):
            print(f"\tTask 1.2's {expected_file} output found.\n")
            if os.path.getsize(expected_file) == 0:
                print(f"\t❗ Task 1.2's {expected_file} output has size zero - please verify it uploaded correctly.\n")
        else:
            print(f"\t❗ Task 1.2's {expected_file} output NOT found. Please check your code.\n")

    print("Finished Task 1.2")
    print("=" * 80)

def main():
    args = sys.argv
    assert len(args) >= 2, "Please provide a task."
    task = args[1]
    valid_tasks = ["all"] + ["task1_" + str(i) for i in range(1, 3)]
    assert task in valid_tasks, \
        f"Invalid task \"{task}\", options are: {valid_tasks}."
    if task == "task1_1":
        verify_task1_1()
    elif task == "task1_2":
        verify_task1_2()
    elif task == "all":
        verify_task1_1()
        verify_task1_2()

if __name__ == "__main__":
    main()

import os
import subprocess


def test_example_script_output():
    # Run the example.py script
    cube_demo_simple = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'basic', 'python_script'))}/cube_ticoi_demo.py"
    # cube_demo_its_live = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'advanced'))}/cube_ticoi_demo_its_live.py"

    result = subprocess.run(["python3", cube_demo_simple], capture_output=True, text=True)
    # Check if the script ran successfully
    assert result.returncode == 0, "The example cube_demo_simple has crashed"

    # result = subprocess.run(["python3", cube_demo_its_live], capture_output=True, text=True)
    # # Check if the script ran successfully
    # assert result.returncode == 0, "The example cube_demo_its_live has crashed"

import os
import subprocess


def test_example_script_output():
    # Run the example.py script
    cube_demo_flag = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples","advanced"))}/cube_ticoi_demo_advanced.py'
    cube_demo_simple = (
        f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))}/cube_ticoi_demo.py'
    )

    print("Testing cube_demo")
    result = subprocess.run(["python", cube_demo_simple], capture_output=True, text=True)
    # Check if the script ran successfully
    assert result.returncode == 0, "The example cube_demo_simple has crashed"

    # print("Testing cube_demo_advanced")
    # result = subprocess.run(["python", cube_demo_flag], capture_output=True, text=True)
    # # Check if the script ran successfully
    # assert result.returncode == 0, "The example cube_demo_advanced has crashed"

import subprocess
import os

def test_example_script_output():
    # Run the example.py script
    cube_demo = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))}/cube_ticoi_demo.py'
    cube_demo_simple = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))}/cube_ticoi_demo_simple.py'
    pixel_demo = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))}/pixel_ticoi_demo.py'

    print('Testing cube_demo')
    result = subprocess.run(['python', cube_demo], capture_output=True, text=True)
    # Check if the script ran successfully
    assert result.returncode == 0, "The example cube_demo has crashed"

    print('Testing cube_simple')
    result = subprocess.run(['python', cube_demo_simple], capture_output=True, text=True)
    # Check if the script ran successfully
    assert result.returncode == 0, "The example cube_demo_simple has crashed"

    print('Testing pixel_demo')
    result = subprocess.run(['python', pixel_demo], capture_output=True, text=True)
    # Check if the script ran successfully
    assert result.returncode == 0, "The example pixel_demo_simple has crashed"
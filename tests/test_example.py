import os
import subprocess


def test_example_script_output():
    # Run the example.py script
    cube_demo_simple = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'basic', 'python_script'))}/cube_ticoi_demo.py"
    pixel_demo_simple = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'basic', 'notebook'))}/pixel_demo_local_ncdata.ipynb"
    pixel_demo_simple_cloud = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'basic', 'notebook'))}/pixel_demo_its_live_on_cloud.ipynb"
    static_areas = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'advanced'))}/stats_in_static_areas.py"
    # cube_demo_its_live = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'advanced'))}/cube_ticoi_demo_its_live.py"

    result = subprocess.run(["python3", cube_demo_simple], capture_output=True, text=True)
    # Check if the script ran successfully
    assert result.returncode == 0, "The example cube_demo_simple has crashed"

    result = subprocess.run(["python3", static_areas], capture_output=True, text=True)
    # Check if the script ran successfully
    assert result.returncode == 0, "The example stats_in_static_areas has crashed"

    result = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", pixel_demo_simple],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"The notebook /pixel_demo_local_ncdata crashed:\n{result.stderr}"

    result = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", pixel_demo_simple_cloud],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"The notebook pixel_demo_its_live_on_cloud crashed:\n{result.stderr}"

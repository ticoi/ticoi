import os
import re

import yaml


def parse_requirements(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    requirements = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            # Using regex to split the package and version constraints
            match = re.match(r"([a-zA-Z0-9_\-]+)([><=!]=?[\d\.]*)?", line)
            if match:
                package = match.group(1)
                version = match.group(2) if match.group(2) else None
                requirements[package] = version

    return requirements


def generate_yaml(requirements, output_file):
    with open(output_file, "w") as file:
        yaml.dump({"dependencies": requirements}, file, default_flow_style=False)


requirements_file = f"{os.path.abspath(os.path.join(os.path.dirname(__file__)))}/requirements.txt"
output_yaml_file = f"{os.path.abspath(os.path.join(os.path.dirname(__file__)))}/ticoi_env.yml"

requirements = parse_requirements(requirements_file)
generate_yaml(requirements, output_yaml_file)

print(f"Generated {output_yaml_file} from {requirements_file}")

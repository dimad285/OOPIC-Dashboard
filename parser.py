import re

def modify_parameters(file_path, output_path, param_updates):
    """
    Modify specified parameters in a structured text file.

    :param file_path: Path to the input file.
    :param output_path: Path to save the modified file.
    :param param_updates: Dictionary of parameter names and new values.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    updated_lines = []
    
    for line in lines:
        match = re.match(r'(\s*)(\w+)\s*=\s*([^/\n]+)', line)
        if match:
            indent, key, value = match.groups()
            if key in param_updates:
                new_value = param_updates[key]
                line = f"{indent}{key} = {new_value}"
                if '//' in line:
                    comment_index = line.index('//')
                    line = line[:comment_index].strip() + ' ' + line[comment_index:]  # Keep comment spacing
                line += '\n'
        updated_lines.append(line)
    
    with open(output_path, 'w') as file:
        file.writelines(updated_lines)

# Example usage
if __name__ == "__main__":
    input_file = "input/Spherical.inp"
    output_file = "input/Spherical_modified.inp"
    parameters_to_change = {
        "d": 0.002,
        "U": 2000,
        "T": 0.05,
        "Qflag": 0
    }
    
    modify_parameters(input_file, output_file, parameters_to_change)
    print(f"Modified file saved as {output_file}")

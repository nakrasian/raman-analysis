import os
import re

metadata_dir = 'metadata-selected'
instructions_file = 'Raman Spectroscopy Instructions.md'

# Define the mapping based on the instructions file
mapping = {
    (1, 2): ("SERS substrate only", "Blank"),
    (3, 6): ("Dora T5", "No Coffee"),
    (7, 10): ("Dora T6", "Coffee"),
    (11, 11): ("SERS substrate only", "Blank"),
    (12, 15): ("Maylette T5", "No Coffee"),
    (16, 19): ("Maylette T6", "Coffee"),
    (20, 24): ("Adrian T5", "No Coffee"),
    (25, 26): ("SERS substrate only", "Blank"),
    (27, 30): ("Adrian T6", "Coffee"),
    (31, 34): ("Allyson T6", "Coffee"),
    (35, 35): ("Coffee", "NA"),
    (36, 39): ("Allyson T5", "No Coffee"),
    (40, 43): ("Nathan T5", "No Coffee"),
    (44, 47): ("Nathan T6", "Coffee"),
    (48, 49): ("SERS substrate only", "Blank"),
    (50, 53): ("1 mmol 4MBA", "4MBA"),
    (54, 57): ("1 mmol 4MBA on quartz", "4MBA"),
    (58, 59): ("1 mmol 4MBA with quartz slide in housing", "4MBA"),
    (60, 61): ("Blank SERS substrate", "Ambient Air"),
    (62, 62): ("Maylette breathing in housing", "Breathing"),
    (63, 63): ("Adrian breathing in housing", "Breathing"),
    (64, 64): ("Blank SERS substrate", "Ambient Air"),
    (65, 65): ("Adrian breathing in housing", "Breathing"),
    (66, 66): ("Nitrogen gas flowing", "Nitrogen"),
    (67, 67): ("Nitrogen gas off", "Nitrogen"),
}

def update_metadata(file_path, label, group):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    content = "".join(lines)
    
    # Check if Label and Group already exist
    has_label = re.search(r'^Label:', content, re.MULTILINE)
    has_group = re.search(r'^Group:', content, re.MULTILINE)
    
    new_lines = []
    found_label = False
    found_group = False
    
    for line in lines:
        if line.startswith('Label:'):
            new_lines.append(f"Label: {label}\n")
            found_label = True
        elif line.startswith('Group:'):
            new_lines.append(f"Group: {group}\n")
            found_group = True
        else:
            new_lines.append(line)
            
    if not found_label:
        if not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
        new_lines.append(f"Label: {label}\n")
    if not found_group:
        if not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
        new_lines.append(f"Group: {group}\n")
        
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

# --- NEW: Dictionary to track how many times we've seen each base label ---
location_counters = {}

for i in range(1, 68):
    file_name = f"Captured_spectra_{i}_metadata.txt"
    file_path = os.path.join(metadata_dir, file_name)
    
    if os.path.exists(file_path):
        # Find which range i falls into
        base_label, target_group = None, None
        for (start, end), (label, group) in mapping.items():
            if start <= i <= end:
                base_label, target_group = label, group
                break
        
        if base_label and target_group:
            # --- NEW LOGIC: Update the counter and format the new label ---
            if base_label not in location_counters:
                location_counters[base_label] = 1
            else:
                location_counters[base_label] += 1
            
            # Combine the base label with the location number
            target_label = f"{base_label} L{location_counters[base_label]}"
            
            update_metadata(file_path, target_label, target_group)
            print(f"Updated {file_name} -> Label: {target_label}")
        else:
            print(f"No mapping for {file_name}")
    else:
        print(f"File {file_path} not found")
import json
import re

def fix_all(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'byzantine_rates' in cell['source']:
            lines = cell['source'].split('\n')
            new_lines = []
            for line in lines:
                if 'byzantine_types' in line:
                    continue  # We will re-insert it
                new_lines.append(line)
                
                if 'byzantine_rates' in line:
                    # Find how much indentation this line has
                    match = re.match(r'^(\s*)', line)
                    indent = match.group(1) if match else ''
                    new_lines.append(indent + "cfg['byzantine_types'] = ['sign_flip']")
                    
            cell['source'] = '\n'.join(new_lines)
            
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_all('baseline/notebook_fast_debug.ipynb')
fix_all('baseline/notebook_partB1_defense_random.ipynb')
fix_all('baseline/notebook_partB2_defense_hier.ipynb')

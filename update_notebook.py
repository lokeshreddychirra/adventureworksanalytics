import json

file_path = r"d:\Lokesh\Projects\adventureworksanalytics\databricks_rag_solution\02_generate_chunks.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modify cell 2 (index 2) - pip install
# The original source was ["%pip install langchain"]
# We want to replace it.
# We need to be careful about which cell it is.
# Based on view_file:
# Cell 0: Markdown "Why do we need chunking?"
# Cell 1: Markdown "Step 1: Install Libraries"
# Cell 2: Code "%pip install langchain"
if nb['cells'][2]['cell_type'] == 'code' and '%pip install langchain' in "".join(nb['cells'][2]['source']):
    nb['cells'][2]['source'] = ["%pip install langchain langchain-community langchain-text-splitters"]

# Modify cell 7 (index 7) - imports
# Based on view_file, it's the cell after "Step 4: Split Text into Chunks" (Cell 6)
# So it should be Cell 7.
# Let's iterate to be safe.
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'from langchain.text_splitter import RecursiveCharacterTextSplitter' in line:
                new_source.append('from langchain_text_splitters import RecursiveCharacterTextSplitter\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

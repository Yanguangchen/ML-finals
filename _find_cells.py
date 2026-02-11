import json
with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if '8.4' in src and 'Deployment considerations' in src:
        print('8.4:', i)
    if '8.5' in src and 'Conclusion' in src:
        print('8.5:', i)
    if 'Bibliography' in src:
        print('Bibliography:', i)

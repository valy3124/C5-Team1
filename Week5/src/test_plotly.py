import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    'UMAP_1': [1,2,3,4],
    'UMAP_2': [1,2,3,4],
    'cluster': ['A', 'A', 'B', 'B'],
    'split': ['train', 'val', 'train', 'val'],
    'filename': ['f1','f2','f3','f4']
})

fig = px.scatter(df, x='UMAP_1', y='UMAP_2', color='cluster', symbol='split')
for t in fig.data:
    print(t.name)

import json

path = "/ghome/group01/C5/vali/C5-Team1/Week3/results/baseline/resnet18_lr0.0005_bs128_cmiqlnrw/captions_history.json"
with open(path, 'r') as f:
    data = json.load(f)

img_name = "VizWiz_train_00011300.jpg"
print(f"EVOLUTION FOR {img_name}:")
for ep in range(1, 13):
    records = data[str(ep)]
    for r in records:
        if r['image_name'] == img_name:
            print(f"Epoch {ep}: {r['prediction']}")

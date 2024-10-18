import json

with open("/data1/sxc/VQ_generation/dataset/prompts_fix.json", "r") as file:
    data = json.load(file)

data = data.items()
data = sorted(data, key=lambda x: x[0])
with open("all_prompts.txt", "w") as f:
    for p, text in data:
        f.write(f"{text}\n")
    


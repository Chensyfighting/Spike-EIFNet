import torch

weight_path = r"D:\productive\selfPythonProgram\EventCamera\SpikeEifnet\pretrained_models\DSEC\DSEC.pth"
# 关键参数 weights_only=False
checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

# 后续解析权重逻辑不变
if isinstance(checkpoint, dict):
    print("顶层字典key：", list(checkpoint.keys()))
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# 打印每层参数信息
total = 0
for name, param in state_dict.items():
    cnt = param.numel()
    total += cnt
    print(f"{name} | shape={param.shape} | params={cnt:,}")
print(f"\n总参数量：{total:,}")
import torch
import sys

DEVICE = 'cuda'
try:
    ifile = sys.argv[1]
except:
    ifile = "example_input"

in_arr = []
for line in open(ifile):
    s = line.strip()
    if s:
        in_arr.append([int(x) for x in s.split()])


t = torch.Tensor(in_arr).to(DEVICE, dtype=torch.int).T

sorted = t.sort(dim=1).values
difference = sorted[0] - sorted[1]
difference.abs_()
print("Part 1:", difference.sum().item())

bins = t[1].bincount(minlength=t[0].max() + 1)
bins *= torch.arange(0, bins.shape[0], device=DEVICE)
print("Part 2:", bins.index_select(dim=0, index=t[0]).sum().item())


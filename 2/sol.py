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
lens = torch.Tensor([len(x) for x in in_arr]).to(DEVICE, dtype=torch.int)
max_len = lens.max().item()
for s in in_arr:
    if len(s) < max_len:
        s += [0] * (max_len - len(s))

t = torch.Tensor(in_arr).to(DEVICE, dtype=torch.int)

def trues(lens):
    max_len = lens.max()
    should_count = torch.arange(0, max_len, device=DEVICE).expand(lens.shape[0], max_len)
    return should_count < lens.unsqueeze(1).expand(-1, max_len)

should_count = trues(lens - 1)

def masked_all(x, mask):
    return x.logical_or(~mask).all(dim=1)

def run(t, mask):
    diffs = t.diff(dim=1)
    samedir = masked_all(diffs >= 0, mask) | masked_all(diffs <= 0, mask)
    absdiffs = diffs.abs()
    diffsize = masked_all(absdiffs >= 1, mask) & masked_all(absdiffs <= 3, mask)
    return samedir & diffsize
validities = run(t, should_count)
print(validities.sum().item())

for i in range(0, max_len):
    cols = torch.arange(0, max_len, device=DEVICE)
    cols = cols[cols != i]
    diffcols = torch.arange(0, max_len - 1, device=DEVICE)
    diffcols = diffcols[diffcols != max(0, i - 1)]
    validities |= run(t[:, cols], should_count[:, diffcols])
print(validities.sum().item())

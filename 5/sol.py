import torch
from torch import tensor
import sys

DEVICE = 'cuda'
try:
    ifile = sys.argv[1]
except:
    ifile = "example_input"

[orderings, updates] = open(ifile).read().split('\n\n', maxsplit=1)
orderings = [[int(y) for y in x.strip().split('|')] for x in orderings.split('\n') if x.strip()]
updates = [torch.tensor([int(y) for y in x.strip().split(',')], device=DEVICE) for x in updates.split('\n') if x.strip()]

adjacencies = torch.zeros((100, 100), dtype=torch.bool, device=DEVICE)

for [before, after] in orderings:
    adjacencies[after, before] = True

def check_update(update):
    return (~torch.triu(adjacencies[update][:,update], diagonal=1).any()).item()

s = torch.tensor(0, device=DEVICE)
incorrect_updates = []
for update in updates:
    if check_update(update):
        s += update[len(update)//2]
    else:
        incorrect_updates.append(update)
print("Part 1:", s.item())

def sort_update(update):
    adj = adjacencies[update][:,update]
    new = torch.zeros_like(update)
    new[adj.sum(dim=1)] = update
    return new

s2 = torch.tensor(0, device=DEVICE)
for update in incorrect_updates:
    new_upd = sort_update(update)
    s2 += new_upd[len(new_upd)//2]

print("Part 2:", s2.item())

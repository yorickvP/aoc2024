import torch
from torch import tensor
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
        in_arr.append(list(map(ord, s)))
XMAS = tensor([88, 77, 65, 83]).to(DEVICE, dtype=torch.uint8)
t = tensor(in_arr).to(DEVICE, dtype=torch.uint8)

def print_matrix(x):
    for row in x:
        rowc = []
        for c in row:
            rowc.append(chr(c) if c > 0 else ".")
        print("".join(rowc))

def strided_count(t, size, stride):
    strided = t.contiguous().as_strided(size=size, stride=stride)
    return strided.eq(XMAS).all(dim=-1).sum()

def horizontal_window(t):
    return strided_count(
        t,
        size=(t.shape[0], t.shape[1] - 3, 4),
        stride=(t.shape[1], 1, 1)
    )

def diagonal_window(t):
    return strided_count(
        t,
        size=(t.shape[0] - 3, t.shape[1] - 3, 4),
        stride=(t.shape[1], 1, t.shape[1] + 1)
    )

straight = \
    horizontal_window(t) + \
    horizontal_window(t.flip(-1)) + \
    horizontal_window(t.T) + \
    horizontal_window(t.T.flip(-1))

diagonal = \
  diagonal_window(t) + \
  diagonal_window(t.flip(1)) + \
  diagonal_window(t.flip(0)) + \
  diagonal_window(t.flip(0).flip(1))
    
print("Part 1", (straight+diagonal).item())

X_SHAPE = tensor([
    [0xFF, 0x00, 0xFF],
    [0x00, 0xFF, 0x00],
    [0xFF, 0x00, 0xFF]
]).to(DEVICE, dtype=torch.uint8)


def xmas_to_mask(shape):
    rows = [[ord(x) for x in n] for n in shape.split('\n') if n]
    return tensor(rows).to(DEVICE, dtype=torch.uint8)

X_MAS = [xmas_to_mask(x) & X_SHAPE for x in [
"""
S S
 A 
M M
"""
,
"""
S M
 A 
S M
"""
,

"""
M S
 A 
M S
"""
,
"""
M M
 A 
S S
"""
]]

def square_window(t):
    return t.contiguous().as_strided(
        size=(t.shape[0] - 2, t.shape[1] - 2, 3, 3),
        stride=(t.shape[1], 1, t.shape[1], 1),
    ) & X_SHAPE

sq = square_window(t)

cnt = torch.zeros(1, dtype=torch.int32, device=DEVICE)
for xmas in X_MAS:
    cnt += sq.eq(xmas).all(dim=(2, 3)).sum().to(torch.int32)

print("Part 2:", cnt.item())

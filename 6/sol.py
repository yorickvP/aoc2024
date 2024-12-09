import torch
from torch import tensor
import sys

DEVICE = 'cpu'
try:
    ifile = sys.argv[1]
except:
    ifile = "example_input"

if len(sys.argv) > 2:
    DEVICE = sys.argv[2]

in_arr = []
    
for line in open(ifile):
    if line.strip():
        in_arr.append(list(map(ord, line.strip())))

t = torch.tensor(in_arr, dtype=torch.uint8, device=DEVICE)
t = torch.nn.functional.pad(t, pad=(1,1,1,1), value=35)

def print_matrix(x):
    for row in x:
        rowc = []
        for c in row:
            rowc.append(chr(c) if c > 0 else ".")
        print("".join(rowc))

START_POS = (t == 94).nonzero()[0]
MAX_POS = torch.tensor((t.shape[0], t.shape[1]), device=DEVICE) - 1

def s(field, pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    if x1 == x2:
        return field[x1,torch.min(y1, y2):torch.max(y1, y2)]
    if y1 == y2:
        return field[torch.min(x1, x2):torch.max(x1, x2),y1]


def find_right(t, pos, dim):
    npos = pos.clone()
    npos[dim] = t.shape[dim]
    npos[dim] = s(t, pos, npos).view(dtype=torch.uint8).argmax() - 1 + pos[dim]
    return npos

@torch.profiler.record_function("move")
def move(visited, t, pos, bit, refl, dim):
    if refl:
        np = MAX_POS - find_right(t, MAX_POS - pos, dim)
    else:
        np = find_right(t, pos, dim)
    s(visited, np, pos).bitwise_or_(1)
    loop = (visited[np[0]][np[1]] & bit)
    visited[np[0]][np[1]] |= bit
    return loop, np

bits = {
    "UP": 2,
    "RIGHT": 4,
    "DOWN": 8,
    "LEFT": 16
}

# print_matrix(t)
def run(pos, t):
    visited = torch.zeros_like(t, dtype=torch.uint8)
    tl = (t == 35)
    tfl = tl.flip((0,1))
    visited[pos[0]][pos[1]] = bits["UP"]
    looped = torch.tensor(0, device=DEVICE)
    while True:
        loop, pos = move(visited, tfl, pos, bits["UP"], True, 0)
        if pos[0] == 1:
            break
        looped += loop
        if looped:
            break
        loop, pos = move(visited, tl, pos, bits["RIGHT"], False, 1)
        if pos[1] == (t.shape[1] - 2):
            break
        looped += loop
        if looped:
            break
        loop, pos = move(visited, tl, pos, bits["DOWN"], False, 0)
        if pos[0] == (t.shape[0] - 2):
            break
        looped += loop
        if looped:
            break
        loop, pos = move(visited, tfl, pos, bits["LEFT"], True, 1)
        if pos[1] == 1:
            break
        looped += loop
        if looped:
            break
    return looped, visited

part1_looped, part1_visited = run(START_POS, t)
print("Part 1", (part1_visited != 0).sum())

def part2():
    loops = 0
    iters = 0
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(1, t.shape[0] - 1):
        for j in range(1, t.shape[1] - 1):
            iters += 1
            if (iters % 100) == 0:
                print("iteration", iters, "of", 130*130)
                iteration = torch.cuda.Event(enable_timing=True)
                iteration.record()
                torch.cuda.synchronize()
                print(start.elapsed_time(iteration))
                start.record()
            if (i,j) == (START_POS[0] - 1, START_POS[1]):
                continue
            cell = t[i][j]
            if cell == ord("."):
                old_cell = cell.clone()
                cell.fill_(ord('#'))
                looped, _ = run(START_POS, t)
                if looped:
                    loops += 1
                    cell.fill_(ord("O"))
                cell.fill_(old_cell)
    return loops

# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ]
# ) as p:
loops = part2()
# p.export_chrome_trace("./test.json")
print("Part 2", loops)

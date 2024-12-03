import torch
import sys
import re

DEVICE = 'cuda'
try:
    ifile = sys.argv[1]
except:
    ifile = "example_input"

in_file = open(ifile).read()
nums = []
enable = 1
# using regex is a bit of a cop-out
for x in re.findall('(mul)\((\d{1,3}),(\d{1,3})\)|(do)\(\)|(don\'t)\(\)', in_file):
    match x:
        case ("mul", a, b, "", ""):
            nums.append([int(a), int(b), enable])
        case ("", "", "", "do", ""):
            enable = 1
        case ("", "", "", "", "don't"):
            enable = 0
        case _:
            raise ValueError(f"invalid match {x}")
    
arr = torch.tensor(nums, dtype=torch.int, device=DEVICE).T
print("part 1:", (arr[0] * arr[1]).sum().item())
print("part 2:", (arr[0] * arr[1] * arr[2]).sum().item())

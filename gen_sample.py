import math

def normalize(rot):
    l = math.sqrt(rot[0]**2 + sum([n**2 for n in rot[1]]))
    return (rot[0]/l, [n/l for n in rot[1]])

def add(f, pos, rot):
    rot = normalize(rot)
    f.write(",".join([str(n) for n in pos + [rot[0]] + rot[1]])+"\n")

with open("sample1.txt", "w") as f:
    for c in range(1000):
        add(f, [math.sin(c/30), 0, 0], (0, [0.0, 0.0, 1.0]))

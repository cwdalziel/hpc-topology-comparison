# generate_hypercube.py
# N must be a power of 2
n = 64  # 64 nodes = 6-dimensional hypercube (2^6)

print("<?xml version='1.0'?>")
print('<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">')
print('<platform version="4.1">')
print('  <zone id="world" routing="Floyd">')

# Hosts
for i in range(n):
    print(f'    <host id="node-{i}.simgrid.org" speed="100Gf"/>')

# Links — two nodes are neighbors if they differ by exactly 1 bit
for i in range(n):
    for bit in range(6):  # 6 dimensions for 64 nodes
        j = i ^ (1 << bit)  # flip bit to get neighbor
        if i < j:  # avoid duplicate links
            print(f'    <link id="link-{i}-{j}" bandwidth="12.5GBps" latency="1us"/>')

# Routes — only define direct neighbor links, Floyd fills in the rest
for i in range(n):
    for bit in range(6):
        j = i ^ (1 << bit)
        if i < j:
            print(f'    <route src="node-{i}.simgrid.org" dst="node-{j}.simgrid.org">')
            print(f'      <link_ctn id="link-{i}-{j}"/>')
            print("    </route>")

print("  </zone>")
print("</platform>")

"""Play a recorded negotiation. No API key, no network, no data download.

    python examples/offline_demo.py
"""

from hdsim.core import replay

for row in replay.available():
    print(f"{row['id']}  {row['members']} members  {row['rounds']} rounds  -> {row['result']}")

print()
print(replay.render(replay.get()))

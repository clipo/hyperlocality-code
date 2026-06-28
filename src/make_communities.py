#!/usr/bin/env python
"""Derive the ahu visual communities from the precomputed intervisibility matrices.

This is the DEM-free step between the (DEM-dependent) viewshed precompute and the
cross-proxy concordance. It reads the committed intervisibility data
(`data/viewshed/ahu_viewshed.json`, produced by `src/ahu_viewshed.py` over the
licensed 50 cm DEM) and writes `data/viewshed/ahu_communities.json`, the per-ahu
visual-community labels that `src/run_concordance.py` consumes.

Two ahu are joined when each can see the other's moai (mutual intervisibility at
each ahu's own moai height); communities are the greedy-modularity partition of
that network, ordered largest-first, with singletons labeled -1. This is the same
construction used in `src/make_viewshed_figure.py`. No DEM required.

Run:  python3 src/make_communities.py
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

VS = Path("data/viewshed/ahu_viewshed.json")
OUT = Path("data/viewshed/ahu_communities.json")


def communities(vs):
    """Mutual-intervisibility graph + greedy-modularity communities.

    Returns (label_per_node, n_communities, n_edges). Matches the partition in
    src/make_viewshed_figure.py and src/viewshed_models.py.
    """
    ahu = vs["ahu"]
    N = len(ahu)
    minh = vs["min_height"]
    heights = [a.get("max_h") for a in ahu]

    def h(j):
        return heights[j] if heights[j] else 4.0

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            ab = minh[i][j] is not None and minh[i][j] <= h(j)
            ba = minh[j][i] is not None and minh[j][i] <= h(i)
            if ab and ba:
                G.add_edge(i, j)

    comms = sorted(nx.community.greedy_modularity_communities(G), key=len, reverse=True)
    label = [-1] * N
    cid = 0
    for c in comms:
        if len(c) >= 2:
            for n in c:
                label[n] = cid
            cid += 1
    n_comm = cid
    return label, n_comm, G.number_of_edges()


def main():
    vs = json.loads(VS.read_text())
    label, n_comm, n_edges = communities(vs)
    ahu = vs["ahu"]
    payload = {
        "ahu": [
            {"site": a["site"], "lat": a["lat"], "lon": a["lon"],
             "community": label[i], "max_h": a.get("max_h")}
            for i, a in enumerate(ahu)
        ],
        "n_comm": n_comm,
        "n_edges": n_edges,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT} ({n_comm} communities, {n_edges} edges, {len(ahu)} ahu)")


if __name__ == "__main__":
    main()

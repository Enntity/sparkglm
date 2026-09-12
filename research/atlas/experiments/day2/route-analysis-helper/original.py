# SPDX-License-Identifier: AGPL-3.0-only

def analyze_routes(records, expert_count, split, expected_owners, expected_layers):
    # ---- validate scalar configuration ----
    def _is_int(x):
        return isinstance(x, int) and not isinstance(x, bool)

    if not _is_int(expert_count) or expert_count <= 0:
        raise ValueError("expert_count must be a positive integer")
    if not _is_int(split) or split < 0 or split > expert_count:
        raise ValueError("split must be an integer in [0, expert_count]")
    if not _is_int(expected_owners) or expected_owners <= 0:
        raise ValueError("expected_owners must be a positive integer")
    if not _is_int(expected_layers) or expected_layers <= 0:
        raise ValueError("expected_layers must be a positive integer")

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("records must be a nonempty list")

    # ---- validate records and gather geometry ----
    rows_geom = None
    topk_geom = None
    # map: group -> owner -> layer -> set of all expert ids (unique over all rows)
    data = {}
    # track seen (group, owner, layer) combos
    seen = set()

    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError("each record must be a dictionary")
        if set(rec.keys()) != {"group", "owner", "layer", "routes"}:
            raise ValueError("record keys must be exactly group, owner, layer, routes")

        group = rec["group"]
        owner = rec["owner"]
        layer = rec["layer"]
        routes = rec["routes"]

        if not isinstance(group, str) or group == "":
            raise ValueError("group must be a nonempty string")
        if not isinstance(owner, str) or owner == "":
            raise ValueError("owner must be a nonempty string")
        if not _is_int(layer) or layer < 0 or layer >= expected_layers:
            raise ValueError("layer out of range")

        if not isinstance(routes, list) or len(routes) == 0:
            raise ValueError("routes must be a nonempty list")

        # geometry consistency
        this_rows = len(routes)
        first_row = routes[0]
        if not isinstance(first_row, list) or len(first_row) == 0:
            raise ValueError("each route row must be a nonempty list")
        this_topk = len(first_row)

        if rows_geom is None:
            rows_geom = this_rows
            topk_geom = this_topk
        else:
            if this_rows != rows_geom:
                raise ValueError("row count must be equal across all records")
            if this_topk != topk_geom:
                raise ValueError("top-k width must be equal across all rows")

        combo = (group, owner, layer)
        if combo in seen:
            raise ValueError("duplicate (group, owner, layer)")
        seen.add(combo)

        if group not in data:
            data[group] = {}
        gmap = data[group]
        if owner not in gmap:
            gmap[owner] = {}
        omap = gmap[owner]
        id_set = omap.get(layer)
        if id_set is None:
            id_set = set()
            omap[layer] = id_set

        for row in routes:
            if not isinstance(row, list) or len(row) == 0:
                raise ValueError("each route row must be a nonempty list")
            if len(row) != this_topk:
                raise ValueError("top-k width must be equal across every row")
            row_seen = set()
            for eid in row:
                if not _is_int(eid) or eid < 0 or eid >= expert_count:
                    raise ValueError("expert id out of range")
                if eid in row_seen:
                    raise ValueError("expert ids must be distinct within a row")
                row_seen.add(eid)
                id_set.add(eid)

    # ---- validate group/owner/layer completeness ----
    for group in data:
        gmap = data[group]
        if len(gmap) != expected_owners:
            raise ValueError("group must have exactly expected_owners distinct owners")
        # same owner set across every layer
        layers_present = None
        for owner in gmap:
            owner_layers = set(gmap[owner].keys())
            if layers_present is None:
                layers_present = owner_layers
            else:
                if owner_layers != layers_present:
                    raise ValueError("owner set must be identical across layers")
        if layers_present != set(range(expected_layers)):
            raise ValueError("missing layer for some owner")
        # one record for every owner/layer combination
        for owner in gmap:
            if set(gmap[owner].keys()) != set(range(expected_layers)):
                raise ValueError("missing (owner, layer) record")

    n_groups = len(data)
    if n_groups * expected_owners * expected_layers != len(records):
        raise ValueError("record count does not match group/owner/layer combinations")

    # ---- compute metrics ----
    def _round(x):
        return float(x)

    out_groups = []
    for group in sorted(data.keys()):
        gmap = data[group]
        owners_sorted = sorted(gmap.keys())
        out_layers = []
        for layer in range(expected_layers):
            per_owner = []
            joint_rank0 = set()
            joint_rank1 = set()
            serial_critical = 0
            owner_unique_visits = 0
            for owner in owners_sorted:
                ids = gmap[owner][layer]
                r0 = set(e for e in ids if e < split)
                r1 = set(e for e in ids if e >= split)
                c0 = len(r0)
                c1 = len(r1)
                per_owner.append({
                    "owner": owner,
                    "rank_unique": [c0, c1],
                })
                joint_rank0 |= r0
                joint_rank1 |= r1
                if c0 > c1:
                    serial_critical += c0
                else:
                    serial_critical += c1
                owner_unique_visits += c0 + c1
            j0 = len(joint_rank0)
            j1 = len(joint_rank1)
            joint_critical = j0 if j0 > j1 else j1
            joint_unique = j0 + j1
            if joint_unique > 0:
                reuse_factor = _round(owner_unique_visits / joint_unique)
            else:
                reuse_factor = 0.0
            if joint_critical > 0:
                critical_count_ratio = _round(serial_critical / joint_critical)
            else:
                critical_count_ratio = 0.0
            out_layers.append({
                "layer": layer,
                "per_owner": per_owner,
                "serial_critical_expert_visits": serial_critical,
                "joint_rank_unique": [j0, j1],
                "joint_critical_experts": joint_critical,
                "owner_unique_visits": owner_unique_visits,
                "joint_unique_experts": joint_unique,
                "reuse_factor": reuse_factor,
                "critical_count_ratio": critical_count_ratio,
            })
        out_groups.append({
            "group": group,
            "owners": owners_sorted,
            "layers": out_layers,
        })

    return {
        "geometry": {
            "rows": rows_geom,
            "topk": topk_geom,
            "expert_count": expert_count,
            "split": split,
            "expected_owners": expected_owners,
            "expected_layers": expected_layers,
        },
        "metric_scope": "expert_counts_not_latency_prediction",
        "groups": out_groups,
    }

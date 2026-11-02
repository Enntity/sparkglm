#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Host-side fabric selection for start.sh; no network or process mutations."""
import ipaddress
import re
import sys

NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')


def inventory(text):
    ports = []
    for line in text.splitlines():
        hca, interface, cidr = line.split()
        if not NAME.fullmatch(hca) or not NAME.fullmatch(interface):
            raise ValueError('invalid device name in fabric inventory')
        address = ipaddress.IPv4Interface(cidr)
        item = (hca, interface, address)
        if item not in ports:
            ports.append(item)
    return ports


def selection(ports, hcas, interface, label):
    names = hcas.split(',') if hcas else []
    if names and (any(not NAME.fullmatch(n) for n in names) or len(names) != len(set(names))):
        raise ValueError(f'{label}_FABRIC_HCA must be a comma-separated list of unique device names')
    if interface and not NAME.fullmatch(interface):
        raise ValueError(f'{label}_FABRIC_INTERFACE is invalid')
    for name in names:
        if not any(p[0] == name for p in ports):
            raise ValueError(f'{label}_FABRIC_HCA: {name} is missing, down, or has no IPv4 address')
    primary = [p for p in ports if (not names or p[0] == names[0]) and (not interface or p[1] == interface)]
    if not primary:
        raise ValueError(f'{label}: no active addressed RDMA interface matches the overrides')
    return primary, names


def same_link(a, b):
    return a[2].network == b[2].network and a[2].ip != b[2].ip


def resolve(left, right, lhca='', rhca='', liface='', riface='', leader_address=''):
    left, right = inventory(left), inventory(right)
    lp, ln = selection(left, lhca, liface, 'LEADER')
    rp, rn = selection(right, rhca, riface, 'WORKER')
    if leader_address:
        address = ipaddress.IPv4Address(leader_address)
        lp = [p for p in lp if p[2].ip == address]
        if not lp:
            raise ValueError('LEADER_ADDRESS is not on the selected leader fabric interface')
    pairs = [(a, b) for a in lp for b in rp if same_link(a, b)]
    if len(pairs) != 1:
        reason = 'no matching' if not pairs else 'ambiguous'
        raise ValueError(f'{reason} fabric discovery ({len(pairs)} candidate pairs); set '
                         'LEADER_FABRIC_HCA/INTERFACE and WORKER_FABRIC_HCA/INTERFACE')
    a, b = pairs[0]
    ln, rn = ln or [a[0]], rn or [b[0]]
    if len(ln) != len(rn):
        raise ValueError('leader and worker must specify the same number of ordered RDMA rails')
    # Every extra rail needs one unique addressed match to the selected peer.
    # Merely being active (or sharing a PCI function) does not prove a peer link.
    for index, (lh, rh) in enumerate(zip(ln, rn)):
        matches = [(x, y) for x in left for y in right if x[0] == lh and y[0] == rh and same_link(x, y)]
        if index == 0:
            matches = [(x, y) for x, y in matches if x == a and y == b]
        if len(matches) != 1:
            raise ValueError(f'RDMA rail {index + 1} ({lh}, {rh}) has {len(matches)} peer matches; '
                             'exclude backchannels and ambiguous addresses')
    return [','.join(ln), a[1], str(a[2].ip), ','.join(rn), b[1], str(b[2].ip)]


def edges(ports):
    """Rails between each pair of Sparks, in subnet order so both ends list them alike."""
    found = {}
    for i, here in enumerate(ports):
        for j, there in enumerate(ports):
            rails = sorted((x[2].network, x, y) for x in here for y in there if i != j and same_link(x, y))
            if rails:
                found.setdefault(i, {})[j] = [(x, y) for _, x, y in rails]
    return found


def ring_order(links, count):
    """Hosts in cable order from host 0, and whether the far end closes the loop."""
    degrees = sorted(len(links.get(host, ())) for host in range(count))
    if degrees not in ([2] * count, [1, 1] + [2] * (count - 2)):
        raise ValueError('the Sparks are cabled as neither a ring nor a chain')
    closed = degrees[0] == 2
    if not closed and len(links[0]) != 1:
        raise ValueError('the first host must be an end of the chain, since rank 0 leads it')
    order = [0]
    while len(order) < count:
        ahead = sorted(set(links[order[-1]]) - set(order))
        if not ahead:
            raise ValueError('the Sparks are cabled as neither a ring nor a chain')
        order.append(ahead[0])
    return order, closed


def topology(inventories):
    """One entry per rank for four Sparks, the first host leading. Sockets run on the
    last rail of the 0-1 and 2-3 links; the hosts' routes carry them to the far ranks."""
    if len(inventories) != 4:
        raise ValueError('topology takes the inventories of four Sparks')
    ports = [inventory(text) for text in inventories]
    links = edges(ports)
    order, closed = ring_order(links, len(ports))
    rails = lambda here, there: ','.join(x[0] for x, _ in links[here][there])
    ranks = []
    for rank, host in enumerate(order):
        left = order[rank - 1] if rank or closed else None
        right = order[(rank + 1) % 4] if rank < 3 or closed else None
        partner = order[rank ^ 1]
        ranks.append({'rank': rank, 'host': host, 'topology': 'ring' if closed else 'chain',
                      'interface': links[host][partner][-1][0][1],
                      'left': '' if left is None else rails(host, left),
                      'right': '' if right is None else rails(host, right)})
    leader = str(links[order[0]][order[1]][-1][0][2].ip)
    return [{**rank, 'leader': leader} for rank in ranks]


def main():
    try:
        if sys.argv[1:2] == ['topology']:
            for rank in topology(sys.argv[2:]):
                print(rank['rank'], rank['host'], rank['topology'], rank['interface'], rank['leader'],
                      rank['left'] or '-', rank['right'] or '-')
            return 0
        print(' '.join(resolve(*sys.argv[1:])))
    except (ValueError, TypeError) as error:
        print(f'fabric: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

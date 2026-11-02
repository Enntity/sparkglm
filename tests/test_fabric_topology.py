# SPDX-License-Identifier: AGPL-3.0-only
"""Four Sparks: how they are cabled is read from their port inventories."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fabric  # noqa: E402

A, B = 'rocep1s0f0 enp1s0f0np0', 'roceP2p1s0f0 enP2p1s0f0np0'
C, D = 'rocep1s0f1 enp1s0f1np1', 'roceP2p1s0f1 enP2p1s0f1np1'
# The bench's real wiring: 01-02, 02-03, 03-04, 04-01, two rails on each link.
RING = [
    f'{A} 10.100.192.2/24\n{C} 10.100.196.1/24\n{B} 10.100.193.2/24\n{D} 10.100.197.1/24',
    f'{A} 10.100.192.1/24\n{C} 10.100.194.1/24\n{B} 10.100.193.1/24\n{D} 10.100.195.1/24',
    f'{A} 10.100.194.2/24\n{C} 10.100.112.1/24\n{B} 10.100.195.2/24\n{D} 10.100.113.1/24',
    f'{A} 10.100.196.2/24\n{C} 10.100.112.2/24\n{B} 10.100.197.2/24\n{D} 10.100.113.2/24',
]
AC, BD = f'{A.split()[0]},{B.split()[0]}', f'{C.split()[0]},{D.split()[0]}'


def without(inventory, *subnets):
    return '\n'.join(l for l in inventory.splitlines() if not any(s in l for s in subnets))


class Topology(unittest.TestCase):
    def test_a_closed_loop_is_a_ring_with_every_rank_on_both_sides(self):
        ranks = fabric.topology(RING)
        self.assertEqual({r['topology'] for r in ranks}, {'ring'})
        self.assertEqual([r['host'] for r in ranks], [0, 1, 2, 3])
        self.assertEqual([(r['left'], r['right']) for r in ranks],
                         [(BD, AC), (AC, BD), (AC, BD), (BD, AC)])

    def test_rails_on_an_edge_match_in_peer_order(self):
        ranks = fabric.topology(RING)
        # 02's right rails reach 03's left rails device for device, by subnet.
        self.assertEqual(ranks[1]['right'].split(','), ['rocep1s0f1', 'roceP2p1s0f1'])
        self.assertEqual(ranks[2]['left'].split(','), ['rocep1s0f0', 'roceP2p1s0f0'])

    def test_the_first_host_leads_and_every_rank_gets_its_socket_interface(self):
        ranks = fabric.topology(RING)
        self.assertEqual({r['leader'] for r in ranks}, {'10.100.193.2'})
        self.assertEqual([r['interface'] for r in ranks],
                         ['enP2p1s0f0np0', 'enP2p1s0f0np0', 'enP2p1s0f1np1', 'enP2p1s0f1np1'])

    def test_rank_order_follows_the_cables_not_the_host_list(self):
        ranks = fabric.topology([RING[0], RING[2], RING[1], RING[3]])
        self.assertEqual([r['host'] for r in ranks], [0, 2, 1, 3])

    def test_an_open_loop_is_a_chain_with_empty_ends(self):
        chain = [without(RING[0], '.196.', '.197.'), RING[1], RING[2], without(RING[3], '.196.', '.197.')]
        ranks = fabric.topology(chain)
        self.assertEqual({r['topology'] for r in ranks}, {'chain'})
        self.assertEqual((ranks[0]['left'], ranks[3]['right']), ('', ''))
        self.assertEqual((ranks[0]['right'], ranks[3]['left']), (AC, BD))

    def test_a_chain_must_start_at_an_end(self):
        chain = [RING[1], without(RING[0], '.196.', '.197.'), RING[2], without(RING[3], '.196.', '.197.')]
        with self.assertRaisesRegex(ValueError, 'end of the chain'):
            fabric.topology(chain)

    def test_a_missing_link_is_refused(self):
        broken = [without(RING[0], '.192.', '.193.'), without(RING[1], '.192.', '.193.', '.194.', '.195.'),
                  without(RING[2], '.194.', '.195.'), RING[3]]
        with self.assertRaisesRegex(ValueError, 'neither a ring nor a chain'):
            fabric.topology(broken)

    def test_it_takes_four_sparks(self):
        with self.assertRaisesRegex(ValueError, 'four'):
            fabric.topology(RING[:2])


if __name__ == '__main__':
    unittest.main()

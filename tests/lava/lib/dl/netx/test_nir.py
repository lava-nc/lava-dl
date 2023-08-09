import logging
import nir
import os
import unittest

from lava.lib.dl import netx


test_folder = os.path.dirname(os.path.abspath(__file__))
root = os.getcwd()
oxford_net_path = os.path.join(
    root, 'tutorials/lava/lib/dl/netx/oxford/Trained/network.net'
)
pilotnet_net_path = os.path.join(
    root, 'tutorials/lava/lib/dl/netx/pilotnet_snn/network.net'
)

logging_level = logging.DEBUG
logging.basicConfig(level=logging_level,
                    format='[lava_nir]:[%(levelname)s] %(message)s')


class TestNetxNIR(unittest.TestCase):
    def test_oxford_lava_to_nir(self) -> None:
        print('oxford: lava -> nir')
        netx.nir_lava.convert_to_nir(oxford_net_path, 'oxford.nir')
        oxford_nir = nir.read('oxford.nir')
        os.remove('oxford.nir')
        self.assertIsNotNone(oxford_nir)

    def test_pilotnet_lava_to_nir(self) -> None:
        print('pilotnet: lava -> nir')
        netx.nir_lava.convert_to_nir(pilotnet_net_path, 'pilotnet.nir')
        pilot_nir = nir.read('pilotnet.nir')
        os.remove('pilotnet.nir')
        self.assertIsNotNone(pilot_nir)

    def test_oxford_nir_to_lava(self) -> None:
        print('oxford: nir -> lava')
        netx.nir_lava.convert_to_nir(oxford_net_path, 'oxford.nir')
        oxford_nir = nir.read('oxford.nir')
        os.remove('oxford.nir')
        oxford_network_nir = netx.nir_lava.nir_graph_to_lava_network(oxford_nir)
        self.assertIsNotNone(oxford_network_nir)

    def test_pilotnet_nir_to_lava(self) -> None:
        print('oxford: nir -> lava')
        netx.nir_lava.convert_to_nir(pilotnet_net_path, 'pilotnet.nir')
        pilot_nir = nir.read('pilotnet.nir')
        os.remove('pilotnet.nir')
        pilot_network_nir = netx.nir_lava.nir_graph_to_lava_network(pilot_nir)
        self.assertIsNotNone(pilot_network_nir)


if __name__ == '__main__':
    unittest.main()

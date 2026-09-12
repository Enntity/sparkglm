# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
import pathlib
import struct
import tempfile
import unittest
import convert

class ConversionTests(unittest.TestCase):
    def fixture(self, path):
        name='model.language_model.layers.45.mlp.experts.0.gate_proj.weight'
        h={name:{'dtype':'BF16','shape':[16,16],'data_offsets':[0,512]},
           'model.language_model.layers.44.norm.weight':{'dtype':'BF16','shape':[4],'data_offsets':[512,520]},
           '__metadata__':{'format':'pt'}}
        raw=json.dumps(h).encode()
        path.write_bytes(struct.pack('<Q',len(raw))+raw+bytes(range(256))*2+b'keepthis')
        return name
    def test_rewrite_keeps_exact_payload_and_creates_standard_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            source=pathlib.Path(root)/'in'; dest=pathlib.Path(root)/'out'
            name=self.fixture(source)
            quant=lambda data,n,k:(bytes([0x12])*(n*k//2),bytes([0x38])*(n*k//16),struct.pack('<f',0.125))
            output,receipts=convert.rewrite(source,dest,quant)
            self.assertEqual(output[name]['shape'],[16,8])
            self.assertEqual(output[name[:-7]+'.weight_scale']['dtype'],'F8_E4M3')
            self.assertEqual(output[name[:-7]+'.weight_scale_2']['shape'],[])
            copied=receipts['model.language_model.layers.44.norm.weight']
            self.assertEqual(copied['output_sha256'],hashlib.sha256(b'keepthis').hexdigest())
            self.assertFalse(copied['converted'])
            self.assertTrue(receipts[name]['converted'])
            self.assertFalse(any(n.endswith('input_scale') for n in output))
            self.assertEqual(dest.read_bytes()[-8:],b'keepthis')
    def test_invalid_geometry_refuses_before_quantizer(self):
        h={'model.language_model.layers.45.mlp.experts.0.gate_proj.weight':{'dtype':'BF16','shape':[16,15],'data_offsets':[0,480]}}
        with self.assertRaisesRegex(ValueError,'geometry'):
            convert.tensor_plan(h)
    def test_invalid_scale_refuses_completion(self):
        with tempfile.TemporaryDirectory() as root:
            source=pathlib.Path(root)/'in'; dest=pathlib.Path(root)/'out'
            self.fixture(source)
            with self.assertRaisesRegex(ValueError,'global scale'):
                convert.rewrite(source,dest,lambda data,n,k:(bytes(n*k//2),bytes(n*k//16),struct.pack('<f',float('nan'))))

if __name__=='__main__': unittest.main()

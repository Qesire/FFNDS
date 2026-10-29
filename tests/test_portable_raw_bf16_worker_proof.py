import unittest
from tools import portable_raw_bf16_worker_proof as m


class PortableProofTests(unittest.TestCase):
    def test_fresh_identity_keeps_same_four_raw_masters(self):
        p = m.draft(); m.validate(p)
        self.assertEqual(p['id'], m.ID)
        self.assertEqual(p['output_HF_relative'], m.OUTPUT)
        self.assertEqual(sum(v['bytes'] for v in p['models']), 69025362058)
        self.assertEqual(p['models'], m.frozen.parent()['models'])
        self.assertEqual(m.RESOURCES['GPU'], 0)

    def test_unbound_profile_or_changed_model_is_rejected(self):
        p = m.draft(); p['status'] = 'FROZEN_REGISTERED'
        with self.assertRaises(ValueError): m.validate(p)
        p['private_worker_config_sha256'] = '0'*64; m.validate(p)
        p['models'][0]['sha256'] = '1'*64
        with self.assertRaises(ValueError): m.validate(p)


if __name__ == '__main__': unittest.main()

import unittest

import torch

from decoder.decoder import Decoder


class ResetWeightsTests(unittest.TestCase):
    def test_reset_restores_initial_state_and_accepts_another_seed(self):
        decoder = Decoder(n_layers=1, emb_dim=8, n_heads=2, context_length=4)
        initial = {k: v.clone() for k, v in decoder.decoder_model.state_dict().items()}
        with torch.no_grad():
            for parameter in decoder.decoder_model.parameters():
                parameter.add_(1)
        decoder.reset_weights()
        for name, value in decoder.decoder_model.state_dict().items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        self.assertEqual(next(decoder.decoder_model.parameters()).device.type, decoder.device.type)
        decoder.reset_weights(seed=456)
        self.assertFalse(torch.equal(decoder.decoder_model.tok_emb.weight, initial['tok_emb.weight']))
        self.assertTrue(torch.isfinite(decoder.decoder_model(torch.tensor([[1, 2]], device=decoder.device))).all())


if __name__ == '__main__':
    unittest.main()

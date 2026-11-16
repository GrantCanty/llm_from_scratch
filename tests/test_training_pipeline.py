import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch.nn.utils.rnn import pad_sequence

from decoder.model import GPTModel, train_model_simple_function
from decoder.utils import calc_loss_loader


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(vocab_size=16, context_length=4, emb_dim=8,
                        n_heads=2, n_layers=1, drop_rate=0., qkv_bias=True)
        notebook = json.loads((Path(__file__).parents[1] / 'main.ipynb').read_text())
        namespace = dict(torch=torch, pad_sequence=pad_sequence, EOT_ID=15,
                         model=SimpleNamespace(GPT_CONFIG=self.cfg))
        source = ''.join(notebook['cells'][5]['source']).split('\ntrain_loader =')[0]
        exec(source, namespace)
        self.collate = namespace['gpt2_collate_fn']

    def test_shift_padding_and_context_bound(self):
        batch = self.collate([{'input_ids': [1, 15, 2, 3, 4, 5]},
                              {'input_ids': [6, 15]}])
        self.assertEqual(batch['input_ids'].tolist(), [[1, 15, 2, 3], [6, 15, 15, 15]])
        self.assertEqual(batch['labels'].tolist(), [[15, 2, 3, 4], [15, -100, -100, -100]])
        self.assertEqual(batch['attention_mask'].tolist(), [[True]*4, [True, False, False, False]])

    def test_causal_and_padding_masks(self):
        model = GPTModel(self.cfg).eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 8, 9]])
        torch.testing.assert_close(model(ids)[:, :2], model(changed)[:, :2])
        mask = torch.tensor([[True, False, True, True]])
        changed = torch.tensor([[1, 9, 3, 4]])
        torch.testing.assert_close(model(ids, mask)[:, 2:], model(changed, mask)[:, 2:])
        with self.assertRaisesRegex(ValueError, 'context length'):
            model(torch.ones(1, 5, dtype=torch.long))

    def test_training_and_evaluation_for_both_batch_formats(self):
        batch = self.collate([{'input_ids': [1, 2, 3, 15]}, {'input_ids': [4, 15]}])
        for loader in ([batch], [(batch['input_ids'], batch['labels'])]):
            model = GPTModel(self.cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            losses = train_model_simple_function(model, loader, loader, optimizer,
                                                 'cpu', 1, 1, 1)
            self.assertTrue(torch.isfinite(torch.tensor(losses[0])).all())
            self.assertEqual(losses[2], [4])
            self.assertTrue(torch.isfinite(torch.tensor(calc_loss_loader(loader, model, 'cpu'))))


if __name__ == '__main__':
    unittest.main()

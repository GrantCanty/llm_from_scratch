import tempfile
import unittest

import torch
from datasets import Dataset

from decoder.packing import CausalLMCollator, iter_packed_windows, pack_dataset
from decoder.model import GPTModel, train_model_simple_function


class ToyTokenizer:
    eot_token = 15

    def encode_batch(self, texts, **kwargs):
        return [[int(token) for token in text.split()] for text in texts]


class PackingTests(unittest.TestCase):
    def test_all_transitions_retained_for_many_lengths_and_contexts(self):
        for context in [1, 2, 4, 8]:
            for lengths in [[], [0], [1], [4], [8], [9], [1, 1, 1], [0, 2, 0, 9, 1]]:
                with self.subTest(context=context, lengths=lengths):
                    docs = [list(range(1, length + 1)) for length in lengths]
                    stream = [token for doc in docs if doc for token in doc + [15]]
                    windows = list(iter_packed_windows(iter(docs), context, 15))
                    actual = [(a, b) for row in windows
                              for a, b in zip(row['input_ids'], row['input_ids'][1:])]
                    self.assertEqual(actual, list(zip(stream, stream[1:])))
                    self.assertTrue(all(len(row['input_ids']) == context + 1 for row in windows[:-1]))
                    self.assertTrue(all(2 <= len(row['input_ids']) <= context + 1 for row in windows))
                    if windows:
                        rebuilt = windows[0]['input_ids'] + [t for row in windows[1:] for t in row['input_ids'][1:]]
                        self.assertEqual(rebuilt, stream)

    def test_short_documents_share_windows_and_long_documents_continue(self):
        windows = list(iter_packed_windows([[1], [2], [3, 4, 5, 6, 7]], 4, 15))
        self.assertEqual([r['input_ids'] for r in windows],
                         [[1, 15, 2, 15, 3], [3, 4, 5, 6, 7], [7, 15]])
        batch = CausalLMCollator(4, 15)(windows)
        self.assertEqual(batch['labels'].tolist(),
                         [[15, -100, 15, -100], [4, 5, 6, 7], [15, -100, -100, -100]])
        self.assertEqual(batch['attention_mask'][0].tolist(), [True] * 4)
        self.assertEqual(batch['attention_mask'][2].tolist(), [True, False, False, False])

    def test_dataset_packing_is_independent_of_tokenization_batch_size(self):
        data = Dataset.from_dict({'text': ['1', '2 3 4 5 6 7 8 9', '', '10', '11 12']})
        with tempfile.TemporaryDirectory() as cache:
            packed = [pack_dataset(data, ToyTokenizer(), 4, tokenization_batch_size=n,
                                   cache_dir=cache) for n in [1, 2, 10]]
            expected = list(iter_packed_windows(ToyTokenizer().encode_batch(data['text']), 4, 15))
            for dataset in packed:
                self.assertEqual(list(dataset), expected)
            repeat = pack_dataset(data, ToyTokenizer(), 4, tokenization_batch_size=2, cache_dir=cache)
            self.assertEqual(list(repeat), expected)
            validation = pack_dataset(Dataset.from_dict({'text': ['13']}), ToyTokenizer(), 4, cache_dir=cache)
            self.assertEqual(list(validation), [{'input_ids': [13, 15], 'document_ids': [0, 0]}])

    def test_real_tokenizer_preserves_document_stream(self):
        import tiktoken
        tokenizer = tiktoken.get_encoding('gpt2')
        texts = ['A short document.', 'Bonjour, le monde! ' * 20, 'Hola.']
        data = Dataset.from_dict({'text': texts})
        expected_stream = [token for ids in tokenizer.encode_batch(texts)
                           for token in ids + [tokenizer.eot_token]]
        with tempfile.TemporaryDirectory() as cache:
            packed = pack_dataset(data, tokenizer, 16, tokenization_batch_size=1, cache_dir=cache)
            rows = list(packed)
        actual_stream = rows[0]['input_ids'] + [token for row in rows[1:] for token in row['input_ids'][1:]]
        self.assertEqual(actual_stream, expected_stream)
        self.assertEqual(sum(len(row['input_ids']) - 1 for row in rows), len(expected_stream) - 1)

    def test_empty_dataset_and_empty_documents(self):
        with tempfile.TemporaryDirectory() as cache:
            for texts in [[], ['', '']]:
                result = pack_dataset(Dataset.from_dict({'text': texts}), ToyTokenizer(), 4, cache_dir=cache)
                self.assertEqual(len(result), 0)
                self.assertEqual(result.column_names, ['input_ids', 'document_ids'])

    def test_invalid_configuration_and_windows(self):
        for context in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                list(iter_packed_windows([[1]], context, 15))
        collate = CausalLMCollator(4, 15)
        for batch in [[], [{'input_ids': [1]}], [{'input_ids': [1] * 6}]]:
            with self.assertRaises(ValueError):
                collate(batch)

    def test_packed_batches_train_with_position_metrics(self):
        rows = list(iter_packed_windows([[1], [2, 3, 4, 5], [6]], 4, 15))
        loader = torch.utils.data.DataLoader(rows, batch_size=2, collate_fn=CausalLMCollator(4, 15))
        cfg = dict(vocab_size=16, context_length=4, emb_dim=8, n_heads=2,
                   n_layers=1, drop_rate=0., qkv_bias=True)
        model = GPTModel(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        results = train_model_simple_function(model, loader, loader, optimizer, 'cpu', 1, 1, 2)
        self.assertTrue(torch.isfinite(torch.tensor(results[0])).all())
        self.assertEqual(results[-1][-1], sum((CausalLMCollator(4, 15)([row])['labels'] != -100).sum().item() for row in rows))
        self.assertEqual(results[3][-1].shape, (4,))

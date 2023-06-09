alphabet_map = {chr(i + 97): (i + 1) for i in range(26)}

model_cfg = {'num_embs': len(alphabet_map),
             'emb_dim': 12,
             'num_blocks': 1,
             'block_cfg': {'num_heads': 1, 'output_dim': 12, 'dropout': 0.1},
             'block_size': 8,
             'input_emb_dropout': 0.1}

train_cfg = {'lr': 1e-2,
             'epochs': 15}

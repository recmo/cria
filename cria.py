#!/usr/bin/env python

from pathlib import Path
import zipfile
import json
import pickle
import struct
import numpy as np
from tqdm import tqdm

from sentencepiece import SentencePieceProcessor

def rms_norm(x):
    return (x / np.sqrt(np.square(x).mean(-1, keepdims=True) + 1e-6))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def mul_col(x, weights):
    # See <https://github.com/facebookresearch/fairscale/blob/3fbde78ffccc28542fa5db9c2581822b29e1fabe/fairscale/nn/model_parallel/layers.py#L218> 
    return np.hstack(list(x @ w.T for w in weights))

def mul_row(x, weights):
    # See <https://github.com/facebookresearch/fairscale/blob/3fbde78ffccc28542fa5db9c2581822b29e1fabe/fairscale/nn/model_parallel/layers.py#L299>
    return sum(s @ w.T for s, w in zip(np.split(x, len(weights), axis=-1), weights))

def load_weights(archive: Path):
    # PyTorch uses an uncompressed zip file containing a single pickle file and raw float data.
    # Since it's uncompressed, we'll memmap it directly from archive to save memory.
    with zipfile.ZipFile(archive, 'r') as zip:
        class Unpickler(pickle.Unpickler):
            def find_class(self, mod_name, name):
                if mod_name == 'torch._utils' and name == '_rebuild_tensor_v2':
                    def rebuild_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                        assert requires_grad == False
                        assert len(backward_hooks) == 0
                        storage = storage[storage_offset:]
                        stride = np.array(stride) * storage.strides
                        return np.lib.stride_tricks.as_strided(storage, size, stride, writeable=False)
                    return rebuild_tensor
                if mod_name == 'torch' and name == 'HalfStorage':
                    return np.float16
                if mod_name == 'torch' and name == 'FloatStorage':
                    return np.float32
                return super().find_class(mod_name, name)
            def persistent_load(self, saved_id):
                (typename, dtype, key, location, count) = saved_id
                name = f'{archive.stem}/data/{key}'
                info = zip.getinfo(name)
                assert info.compress_type == zipfile.ZIP_STORED and info.compress_size == info.file_size
                assert info.file_size == count * dtype(0).itemsize
                with open(archive, 'br') as f:
                    f.seek(info.header_offset)
                    (header, _, n_name, n_extra) = struct.unpack('<4s22shh', f.read(30))
                    assert header == b'PK\x03\x04'
                    content_offset = info.header_offset + 30 + n_name + n_extra
                data = np.memmap(zip.fp, dtype=dtype, mode='r', offset=content_offset, shape=(count,))
                return data
        with zip.open(f'{archive.stem}/data.pkl') as data_pkl:
            data = Unpickler(data_pkl).load()
    return data

# @profile
def main(
    prompt: str = "I believe the meaning of life is",
    n_tokens_to_generate: int = 50,
    model_size: str = "30B",
    models_dir: str = "models",
    max_seq_len: int = 512,
    temperature: float = 0.8, # TODO
):
    # Load Tokenizer
    tokenizer = SentencePieceProcessor(model_file=f"{models_dir}/tokenizer.model")

    # Load sharded model
    model_dir = Path(models_dir) / model_size
    with open(model_dir / 'params.json') as f:
        params = json.load(f)
    print(params)
    head_dim = params['dim'] // params['n_heads']
    sharded = ['tok_embeddings', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3', 'output']
    data = {}
    for archive in sorted(model_dir.glob('*.pth')):
        shard = load_weights(archive)
        for (key, value) in shard.items():
            key = key.replace('.weight', '')
            if key.split('.')[-1] in sharded:
                if key not in data:
                    data[key] = []
                data[key].append(value)
            else:
                #if key in data:
                #    assert (data[key] == value).all()
                data[key] = value
    # for (key, value) in data.items():
    #     print(key, [value.shape for value in value] if type(value) == list else value.shape)
    
    # Compute freq_cis
    freqs = np.logspace(0, 1.0, base=1e-4, num=head_dim // 2, endpoint=False)
    freqs_cis = np.exp(1j * np.outer(np.arange(2 * max_seq_len), freqs)).astype(np.complex64)

    # Tokenize prompt
    tokens = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    print(f"Encoded \"{prompt}\" to {tokens}")

    cache_k = [None] * params['n_layers']
    cache_v = [None] * params['n_layers']
    prev_pos = 0
    for cur_pos in range(len(tokens), len(tokens) + n_tokens_to_generate):

        # Embed tokens
        # See <https://github.com/facebookresearch/fairscale/blob/3fbde78ffccc28542fa5db9c2581822b29e1fabe/fairscale/nn/model_parallel/layers.py#L152>
        token_view = tokens[prev_pos:cur_pos]
        h = np.hstack(list(d[token_view,:].astype(np.float32) for d in data['tok_embeddings']))
        seq_len = cur_pos - prev_pos
        f = freqs_cis[prev_pos:cur_pos].reshape(-1, 1, head_dim // 2)

        for layer in tqdm(range(params['n_layers'])):

            # QKV projections
            wa = data[f'layers.{layer}.attention_norm']
            wq = data[f'layers.{layer}.attention.wq']
            wk = data[f'layers.{layer}.attention.wk']
            wv = data[f'layers.{layer}.attention.wv']
            wo = data[f'layers.{layer}.attention.wo']
            shape = (-1, params['n_heads'], head_dim)
            xn = rms_norm(h) * wa
            xq = mul_col(xn, wq).reshape(shape)
            xk = mul_col(xn, wk).reshape(shape)
            xv = mul_col(xn, wv).reshape(shape)

            # Rotary embedding
            xq = (xq.view(dtype=np.complex64) * f).view(dtype=np.float32)
            xk = (xk.view(dtype=np.complex64) * f).view(dtype=np.float32)

            # Cache
            if prev_pos == 0:
                cache_k[layer] = xk
                cache_v[layer] = xv
            else:
                xk = cache_k[layer] = np.concatenate((cache_k[layer], xk), axis=0)
                xv = cache_v[layer] = np.concatenate((cache_v[layer], xv), axis=0)
            # print("xk = ", xk.shape, xk)

            # Attention
            scores = np.matmul(xk, xq, axes=[(0,2),(2,0),(2,1)]) / np.sqrt(head_dim)
            if seq_len > 1:
                mask = -1e10 * (1 - np.tri(seq_len))
                scores += mask
            scores = softmax(scores)
            output = np.matmul(scores, xv, axes=[(1,2), (0,2), (0,2)]).reshape(-1, params['dim'])
            # TODO: Is np.hstack correct here? I think this should be sharded over input and summed.
            h += mul_row(output, wo)

            # Feed forward neural network with SiLU
            wn = data[f'layers.{layer}.ffn_norm']
            w1 = data[f'layers.{layer}.feed_forward.w1']
            w2 = data[f'layers.{layer}.feed_forward.w2']
            w3 = data[f'layers.{layer}.feed_forward.w3']
            xn = rms_norm(h) * wn
            x1 = mul_col(xn, w1)
            x3 = mul_col(xn, w3)
            x = ((x1 / (1.0 + np.exp(-x1))) * x3)
            h += mul_row(x, w2)
            
        # Output norm
        wn = data['norm']
        wo = data['output']
        h = rms_norm(h) * wn
        logits = mul_col(h[-1, :], wo)

        # Select next token
        # TODO: Temperature
        tokens.append(int(np.argmax(logits)))

        prev_pos = cur_pos

        # Untokenize result
        result = tokenizer.decode(tokens)
        print(result)

if __name__ == "__main__":
    import fire
    fire.Fire(main)

#!/usr/bin/env python

import zipfile
import json
import pickle
import struct
import numpy as np

from sentencepiece import SentencePieceProcessor

def rms_norm(x):
    return (x / np.sqrt(np.square(x).mean(-1, keepdims=True) + 1e-6))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def main(
    prompt: str = "I believe the meaning of life is",
    n_tokens_to_generate: int = 5,
    model_size: str = "7B",
    models_dir: str = "models",
    max_seq_len: int = 512,
):
    # Load Tokenizer
    tokenizer = SentencePieceProcessor(model_file=f"{models_dir}/tokenizer.model")
    # print(f"vocab_size: {tokenizer.vocab_size()}")
    # print(f"bos_id: {tokenizer.bos_id()}")
    # print(f"eos_id: {tokenizer.eos_id()}")
    # print(f"pad_id: {tokenizer.pad_id()}")

    # Load model
    # PyTorch uses an uncompressed zip file containing a single pickle file and raw float data.
    # Since it's uncompressed, we'll memmap it directly from archive to save memory.
    with open(f'{models_dir}/{model_size}/params.json') as f:
        params = json.load(f)
    archive = f'{models_dir}/{model_size}/consolidated.00.pth'
    with zipfile.ZipFile(archive, 'r') as zip:
        class Unpickler(pickle.Unpickler):
            def find_class(self, mod_name, name):
                if mod_name == 'torch._utils' and name == '_rebuild_tensor_v2':
                    def rebuild_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                        assert storage_offset == 0
                        assert requires_grad == False
                        assert len(backward_hooks) == 0
                        stride = np.array(stride) * storage.strides
                        return np.lib.stride_tricks.as_strided(storage, size, stride, writeable=False)
                    return rebuild_tensor
                if mod_name == 'torch' and name == 'HalfStorage':
                    return np.half
                return super().find_class(mod_name, name)
            def persistent_load(self, saved_id):
                (typename, dtype, key, location, count) = saved_id
                name = f'consolidated/data/{key}'
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
        with zip.open('consolidated/data.pkl') as data_pkl:
            data = Unpickler(data_pkl).load()

    print(params)
    for (key, value) in data.items():
        print(key, value.shape)
    head_dim = params['dim'] // params['n_heads']
    
    # Compute freq_cis
    freqs = np.logspace(0, 1.0, base=1e-4, num=head_dim // 2, endpoint=False)
    freqs_cis = np.exp(1j * np.outer(np.arange(2 * max_seq_len), freqs)).astype(np.complex64)

    # Tokenize prompt
    tokens = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    # tokens += [tokenizer.pad_id()] * n_tokens_to_generate
    print(f"Encoded \"{prompt}\" to {tokens}")

    # Embed tokens
    h = data['tok_embeddings.weight'][tokens,:].astype(np.float32)
    print("h = ", h.shape, h)

    start_pos = 0
    seq_len = 8
    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
    # print("freqs_cis = ", freqs_cis.shape, freqs_cis)

    for layer in range(params['n_layers']):
        print("layer = ", layer)

        # QKV projections
        wa = data[f'layers.{layer}.attention_norm.weight']
        wq = data[f'layers.{layer}.attention.wq.weight']
        wk = data[f'layers.{layer}.attention.wk.weight']
        wv = data[f'layers.{layer}.attention.wv.weight']
        wo = data[f'layers.{layer}.attention.wo.weight']
        shape = (-1, params['n_heads'], head_dim)
        xn = rms_norm(h) * wa
        xq = (xn @ wq.T).reshape(shape)
        xk = (xn @ wk.T).reshape(shape)
        xv = (xn @ wv.T).reshape(shape)

        # Rotary embedding
        f = freqs_cis.reshape(-1, 1, xq.shape[-1] // 2)
        xq = (xq.view(dtype=np.complex64) * f).view(dtype=np.float32)
        xk = (xk.view(dtype=np.complex64) * f).view(dtype=np.float32)

        # Attention
        scores = np.matmul(xk, xq, axes=[(0,2),(2,0),(2,1)]) / np.sqrt(head_dim)
        #if layer == 0:
        mask = -1e10 * (1 - np.tri(seq_len))
        scores += mask
        scores = softmax(scores)
        output = np.matmul(scores, xv, axes=[(1,2), (0,2), (0,2)]).reshape(-1, params['dim'])
        h += output @ wo.T

        # Feed forward neural network
        wn = data[f'layers.{layer}.ffn_norm.weight']
        w1 = data[f'layers.{layer}.feed_forward.w1.weight']
        w2 = data[f'layers.{layer}.feed_forward.w2.weight']
        w3 = data[f'layers.{layer}.feed_forward.w3.weight']
        xn = rms_norm(h) * wn
        x1 = xn @ w1.T
        h += ((x1 / (1.0 + np.exp(-x1))) * (xn @ w3.T)) @ w2.T

        print("h = ", h.shape, h)

    # Untokenize result
    #result = tokenizer.decode(tokens)
    #print(result)

if __name__ == "__main__":
    import fire
    fire.Fire(main)

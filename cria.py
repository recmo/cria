#!/usr/bin/env python

import zipfile
import json
import pickle
import struct
import numpy as np

from sentencepiece import SentencePieceProcessor

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = np.arange(end, device=freqs.device)  # type: ignore
    freqs = np.outer(t, freqs).float()  # type: ignore
    freqs_cis = np.polar(np.ones_like(freqs), freqs)  # complex64
    return freqs_cis

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
    # PyTorch uses an uncompressed zip file containing a single pickle file and raw tensor data.
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
    # for (key, value) in data.items():
    #     print(key, value.shape)
    dim = params['dim']
    
    # Compute freq_cis
    theta = 10000.0
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    print(freqs)
    t = np.arange(max_seq_len * 2)
    freqs = np.outer(t, freqs)
    print(freqs)
    freqs_cis = np.exp(freqs * 1j)
    print(freqs_cis)


    # freq_cis = precompute_freqs_cis(params['dim'] // params['n_heads'], )


    # Tokenize prompt
    tokens = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    # tokens += [tokenizer.pad_id()] * n_tokens_to_generate
    print(f"Encoded \"{prompt}\" to {tokens}")

    # Embed tokens
    tokens = data['tok_embeddings.weight'][tokens,:]
    print(tokens.shape, tokens)
    freqs_cis = data['freqs_cis.weight']

    # Untokenize result
    #result = tokenizer.decode(tokens)
    #print(result)

if __name__ == "__main__":
    import fire
    fire.Fire(main)

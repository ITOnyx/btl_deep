"""VRAM Verification Script for InternVL2-2B and Auxiliary Models.
Tests FP16 / 8-bit / 4-bit loading on GPU and measures peak VRAM / OOM behavior.
"""

import os
import sys
import gc
import torch

# Fix Windows PostgreSQL SSL issue if present
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("SSL_CERT_FILE", None)


def get_vram_info():
    if not torch.cuda.is_available():
        return {"cuda": False}
    free_b, total_b = torch.cuda.mem_get_info(0)
    alloc_b = torch.cuda.memory_allocated(0)
    res_b = torch.cuda.memory_reserved(0)
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "total_gb": round(total_b / (1024**3), 3),
        "free_gb": round(free_b / (1024**3), 3),
        "allocated_gb": round(alloc_b / (1024**3), 3),
        "reserved_gb": round(res_b / (1024**3), 3),
    }


def test_load_model(model_id="OpenGVLab/InternVL2-2B", mode="8bit"):
    print(f"=== Testing Load on GPU: {model_id} (Mode: {mode}) ===")
    vram_before = get_vram_info()
    print(f"Initial VRAM: Total {vram_before['total_gb']} GB, Free {vram_before['free_gb']} GB")

    from transformers import AutoModel, BitsAndBytesConfig

    result = {
        "model_id": model_id,
        "mode": mode,
        "initial_vram": vram_before,
        "success": False,
        "error": None,
        "peak_vram_gb": 0.0,
    }

    try:
        torch.cuda.reset_peak_memory_stats(0)
        kwargs = {"trust_remote_code": True, "device_map": "cuda:0"}

        if mode == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif mode == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif mode == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif mode == "cpu":
            kwargs["device_map"] = "cpu"
            kwargs["torch_dtype"] = torch.float32

        print(f"-> Loading model with kwargs: {kwargs}...")
        model = AutoModel.from_pretrained(model_id, **kwargs)
        peak_b = torch.cuda.max_memory_allocated(0)
        peak_gb = round(peak_b / (1024**3), 3)
        print(f"✅ Success! Peak VRAM allocated: {peak_gb} GB")
        result["success"] = True
        result["peak_vram_gb"] = peak_gb
        del model
    except torch.cuda.OutOfMemoryError as oom:
        print(f"❌ CUDA OutOfMemoryError: {oom}")
        result["error"] = "CUDA OutOfMemoryError"
    except Exception as e:
        print(f"❌ Error: {e}")
        result["error"] = str(e)
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "OpenGVLab/InternVL2-2B"
    mode = sys.argv[2] if len(sys.argv) > 2 else "8bit"
    res = test_load_model(model, mode)
    print("Result:", res)

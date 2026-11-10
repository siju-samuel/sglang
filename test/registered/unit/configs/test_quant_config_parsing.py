import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")


class TestQuantLogString(CustomTestCase):
    def test_qwen_fp8_config(self):
        # Example from Qwen/Qwen3-4B-Thinking-2507-FP8
        quant_config = {
            "activation_scheme": "dynamic",
            "modules_to_not_convert": ["lm_head"],
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }

        # Create a raw instance
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=fp8, fmt=e4m3"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test Qwen FP8] Result: {result}")
        self.assertEqual(result, expected)

    def test_llama_gptq_int4_config(self):
        # Example from hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
        quant_config = {"bits": 4, "quant_method": "gptq", "group_size": 128}
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=gptq, bits=4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test Llama GPTQ] Result: {result}")
        self.assertEqual(result, expected)

    def test_awq_config(self):
        quant_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
        }
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=awq, bits=4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test AWQ] Result: {result}")
        self.assertEqual(result, expected)

    def test_modelopt_nvfp4(self):
        quant_config = {"quant_method": "modelopt_fp4", "quant_algo": "NVFP4"}
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=modelopt_fp4, quant_algo=NVFP4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test ModelOpt] Result: {result}")
        self.assertEqual(result, expected)

    def test_no_quant_config(self):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=None)

        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test No Quant] Result: {result}")
        self.assertIsNone(result)


class TestCompressedTensorsMxfp8Routing(CustomTestCase):
    """compressed-tensors MXFP8 must reach the native `mxfp8` config.

    llm-compressor and AutoRound write MXFP8 as compressed-tensors with a
    group-32 float8 scheme. The compressed-tensors scheme table has no MX
    entry, so without this translation the checkpoint raises "No
    compressed-tensors compatible scheme was found" at model load.
    """

    @staticmethod
    def make_config(quant_config):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(quantization_config=quant_config)
        return model_config

    @staticmethod
    def mxfp8_checkpoint(group_size=32, **overrides):
        # Shape of intel-ai/*-MXFP8-CT-AutoRound.
        config = {
            "quant_method": "compressed-tensors",
            "format": "mxfp8-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "group",
                        "group_size": group_size,
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "group",
                        "group_size": group_size,
                        "dynamic": True,
                    },
                }
            },
            "ignore": ["lm_head", "model.layers.0.mlp.gate"],
        }
        config.update(overrides)
        return config

    def test_mxfp8_is_routed_to_the_native_config(self):
        parsed = self.make_config(self.mxfp8_checkpoint())._parse_quant_hf_config()

        self.assertEqual(parsed["quant_method"], "mxfp8")
        self.assertEqual(parsed["weight_block_size"], [1, 32])
        self.assertEqual(parsed["scale_fmt"], "ue8m0")
        self.assertEqual(parsed["activation_scheme"], "dynamic")
        self.assertEqual(
            parsed["modules_to_not_convert"], ["lm_head", "model.layers.0.mlp.gate"]
        )

    def test_group_size_is_carried_into_the_weight_block(self):
        parsed = self.make_config(
            self.mxfp8_checkpoint(group_size=64)
        )._parse_quant_hf_config()

        self.assertEqual(parsed["weight_block_size"], [1, 64])

    def test_fp8_kv_cache_scheme_is_carried(self):
        parsed = self.make_config(
            self.mxfp8_checkpoint(kv_cache_scheme={"type": "float", "num_bits": 8})
        )._parse_quant_hf_config()

        self.assertEqual(parsed["kv_cache_quant_algo"], "FP8")

    def test_mixed_group_sizes_are_rejected(self):
        checkpoint = self.mxfp8_checkpoint()
        checkpoint["config_groups"]["group_1"] = {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "group",
                "group_size": 64,
                "symmetric": True,
                "dynamic": False,
            },
            "input_activations": None,
        }

        with self.assertRaisesRegex(ValueError, "mixes weight group sizes"):
            self.make_config(checkpoint)._parse_quant_hf_config()

    def test_underscore_method_spelling_is_accepted(self):
        parsed = self.make_config(
            self.mxfp8_checkpoint(quant_method="compressed_tensors")
        )._parse_quant_hf_config()

        self.assertEqual(parsed["quant_method"], "mxfp8")

    def test_other_compressed_tensors_formats_are_untouched(self):
        """MXFP4 and W4A16 keep their own paths; only MXFP8 is translated."""
        for fmt in ("mxfp4-pack-quantized", "pack-quantized", "float-quantized"):
            with self.subTest(format=fmt):
                parsed = self.make_config(
                    self.mxfp8_checkpoint(format=fmt)
                )._parse_quant_hf_config()

                self.assertEqual(parsed["quant_method"], "compressed-tensors")
                self.assertNotIn("scale_fmt", parsed)

    def test_modelopt_mxfp8_still_parses(self):
        """The ModelOpt branch and this one must not shadow each other."""
        parsed = self.make_config(
            {"quant_algo": "MXFP8", "quant_method": "modelopt", "group_size": 32}
        )._parse_quant_hf_config()

        self.assertEqual(parsed["quant_method"], "mxfp8")
        self.assertEqual(parsed["weight_block_size"], [1, 32])


if __name__ == "__main__":
    unittest.main()

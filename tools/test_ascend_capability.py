from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.ascend_capability import SOC_RE, normalize_soc, read_capabilities


SAMPLE = """\
[version]
SoC_version=Ascend910B1
Short_SoC_version=Ascend910B
AIC_version=AIC-C-220
CCEC_AIC_version=dav-c220-cube
CCEC_AIV_version=dav-c220-vec
NpuArch=2201
[SoCInfo]
cube_core_cnt=24
vector_core_cnt=48
ai_cpu_cnt=6
memory_size=68719476736
l2_size=201326592
core_type_list=CubeCore,VectorCore
cube_vector_combine=split
[AICoreSpec]
cube_freq=1850
vec_calc_size=128
l0_a_size=65536
l0_b_size=65536
l0_c_size=131072
l1_size=524288
ub_size=196608
ubblock_size=32
ubbank_size=4096
ubbank_num=64
ubbank_group_num=16
[DtypeMKN]
Default=16,16,16
DT_INT8=16,32,16
DT_INT4=16,64,16
[AICoreMemoryRates]
ddr_rate=32
l2_rate=110
"""


class AscendCapabilityTests(unittest.TestCase):
    def test_soc_normalization_requires_an_exact_product(self) -> None:
        self.assertEqual(normalize_soc("Ascend 910B1"), "Ascend910B1")
        self.assertEqual(normalize_soc("910B4-1"), "Ascend910B4-1")
        self.assertEqual(normalize_soc("910B"), "")
        self.assertIsNone(SOC_RE.search("Ascend 910B\n1 device online"))

    def test_platform_config_yields_generation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = (
                Path(temp_dir)
                / "cann-8.5.0"
                / "aarch64-linux"
                / "data"
                / "platform_config"
                / "Ascend910B1.ini"
            )
            config.parent.mkdir(parents=True)
            config.write_text(SAMPLE, encoding="utf-8")
            payload = read_capabilities(config)
        self.assertEqual(payload["identity"]["soc_version"], "Ascend910B1")
        self.assertEqual(payload["identity"]["npu_arch"], 2201)
        self.assertEqual(payload["topology"]["aic_cores"], 24)
        self.assertEqual(payload["topology"]["aiv_cores"], 48)
        self.assertEqual(payload["topology"]["mix_block_dim"], 24)
        self.assertEqual(payload["memory_bytes"]["ub_per_core"], 196608)
        self.assertEqual(payload["compute"]["dtype_mkn"]["default"], "16,16,16")
        peaks = payload["compute"]["dense_peak"]
        self.assertEqual(peaks["fp16_cube_tflops"], 363.7248)
        self.assertEqual(peaks["bf16_cube_tflops_if_fp16_issue_rate"], 363.7248)
        self.assertIsNone(peaks["fp32_cube_tflops"])
        self.assertEqual(peaks["int8_cube_tops"], 727.4496)
        self.assertEqual(peaks["int4_cube_tops_if_one_mmad_per_cycle"], 1454.8992)
        self.assertEqual(peaks["fp16_vector_add_tera_ops_s"], 11.3664)
        self.assertEqual(
            peaks["fp32_vector_add_tera_ops_s_if_one_repeat_per_cycle"],
            5.6832,
        )
        self.assertIn("unknown", peaks["confidence"]["vector_fma"])
        bandwidth = payload["memory_bandwidth"]
        self.assertEqual(bandwidth["raw_platform_rate_fields"]["ddr_rate"], "32")
        self.assertEqual(bandwidth["raw_platform_rate_fields"]["l2_rate"], "110")
        self.assertIn("does not document unit", bandwidth["raw_platform_rate_semantics"])
        self.assertIsNone(bandwidth["cann_msprof_roofline_gm_rw_tb_s"])
        self.assertIsNone(bandwidth["cann_msprof_roofline_l2_rw_tb_s"])
        self.assertIsNone(bandwidth["physical_hbm_interface_peak_tb_s"])
        self.assertEqual(payload["toolchain"]["cann"], "8.5.0")


if __name__ == "__main__":
    unittest.main()

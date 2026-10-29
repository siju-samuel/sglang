"""Shared fakes for the gigakernel unit tests: one spec, one model, server arguments."""

from types import SimpleNamespace

import torch
from msgspec.structs import replace

from sglang.kernels.plugin_kernels.intel import spec as registry
from sglang.kernels.plugin_kernels.intel.spec import (
    GigakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
    Step,
)
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0, stage="base-a", runner_config="cpu", disabled="gigakernel test fakes"
)

ARCH, ISA = "FakeGigaForCausalLM", "bmg_g21"


class FakeBuilt:
    """Covers steps of 1-4 rows and records the steps it ran."""

    def __init__(self, widths):
        self.widths = tuple(w for w in widths if w <= 4)
        self.steps, self.views = [], []

    def summary(self):
        return "fake"

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        rows = int(input_ids.shape[0])
        return Step(mode="decode", rows=rows) if rows in self.widths else None

    def forward(self, step, original, input_ids, positions, forward_batch, **kw):
        self.steps.append(step.rows)
        self.views.append(step.view)
        return "gigakernel"


def _build(model, hf_config, context):
    model.context = context
    return FakeBuilt(context.widths)


SPEC = GigakernelSpec(
    name="fake",
    requires=Requirements(isas=frozenset({ISA}), tp_sizes=frozenset({4})),
    features=frozenset(),
    numerics=Numerics(kind=NumericsKind.BITWISE),
    max_rows=8,
    build=_build,
    api_version=registry.GIGAKERNEL_API_VERSION,
)
# an empty field accepts any value
ANY_TP = replace(SPEC, requires=replace(SPEC.requires, tp_sizes=frozenset()))
# builds no width at all
NO_WIDTH = replace(
    SPEC, build=lambda model, hf, ctx: SimpleNamespace(widths=(), summary=str)
)


class Lm(torch.nn.Module):
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return "standard"


def batch(rows):
    return SimpleNamespace(batch_size=rows, input_ids=torch.zeros(rows))


def server_args(**kw):
    """Arguments as the resolution step sees them, with the graph config resolved."""
    defaults = {
        "model_path": "dummy",
        "tp_size": 4,
        "cuda_graph_config": default_cuda_graph_config(),
    }
    return ServerArgs(**{**defaults, **kw})

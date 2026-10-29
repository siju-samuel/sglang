# SPDX-License-Identifier: Apache-2.0
"""SGLang's binding to the Intel gigakernel plugins behind --enable-intel-gigakernel.

It checks a plugin's spec against the resolved configuration at startup, builds it
once the model is prepared, and installs the whole-step dispatch on the model. It is
also the one place a gigakernel reads SGLang's runtime state: the KV pool and the
attention metadata of the running forward. A spec's build gets a
``GigakernelBinding`` as ``BuildContext.binding``; a covered forward gets the
running forward's ``ForwardView`` as ``Step.view``. Everything else a gigakernel
touches is the model it replaces, SGLang's kernel ops, and the reads listed below.
"""

from __future__ import annotations

import functools
import logging
import time
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec
import torch
import torch.distributed as dist

from sglang.kernels.plugin_kernels.intel.spec import (
    ENTRY_POINT,
    FLAG,
    BuildContext,
    BuiltGigakernel,
    Feature,
    GigakernelSpec,
    Step,
    find_spec,
    registered_architectures,
    spec_registered,
)
from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.plugins import GENERAL_PLUGINS_GROUP
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_exec,
    get_parallel,
    get_spec,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

logger = logging.getLogger(__name__)

# Called by a gigakernel, tolerated until offered as data (its contract test pins
# both lists).
TOLERATED_CALLABLES = (
    "ForwardView.attention",  # the attention backend's own forward
    "sglang.kernels.ops",  # SGLang's kernel ops
    "the replaced model's modules, weights and quant methods",
)
# Written into SGLang's objects, tolerated until SGLang owns these buffers.
TOLERATED_WRITES = ()
# Read, tolerated: configuration, not runtime state.
TOLERATED_READS = (
    "runtime_context getters (get_parallel, get_exec, get_spec, get_platform)",
    "capture-mode queries (get_is_capture_mode, is_in_breakable_cuda_graph)",
)


class ForwardView:
    """The running forward's attention data, read from its attention backend; built
    at capture for a graph step, so its tensors are the ones the graph replays."""

    def __init__(self, backend: Any):
        self._backend = backend
        self.metadata = backend.forward_metadata
        self.kv_pool = backend.token_to_kv_pool

    @property
    def attention(self) -> Callable[..., Any]:
        return self._backend.forward


class GigakernelBinding:
    """Built at install, before the KV pools exist; read at build or forward. Each
    source is sealed at the first read that finds it; ``assert_live`` (after the
    decode graphs are captured) raises once one was reallocated."""

    def __init__(self, *, model: torch.nn.Module):
        self._model = model
        self._sealed: dict[str, tuple] = {}

    def forward_view(self) -> ForwardView:
        """The view of the forward running now."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        view = ForwardView(get_attn_backend())
        self._seal("kv_pool", view.kv_pool)
        return view

    def assert_live(self) -> None:
        current = {"kv_pool": self._current_pool}
        for name, sealed in self._sealed.items():
            if _identity(current[name]()) != sealed:
                raise RuntimeError(
                    f"gigakernel binding: {name} was reallocated after its first read"
                )

    def _seal(self, name: str, value: Any) -> None:
        self._sealed.setdefault(name, _identity(value))

    def _current_pool(self) -> Any:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        return get_attn_backend().token_to_kv_pool


def _identity(value: Any) -> tuple:
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), tuple(value.shape), value.dtype)
    return (id(value),)


PLUGIN_URL = "https://github.com/sgl-project/sgl-intel-giga"

# Leave time for peers to abort before the process watchdog fires.
MAX_DEADLINE_S, MIN_DEADLINE_S = 60.0, 5.0

# torch.xpu reports the SYCL device architecture as its integer enum value
# (sycl/ext/oneapi/experimental/device_architecture.def, oneAPI 2026.1).
_SYCL_ARCHITECTURES = {
    0x000000030F000700: "pvc",
    0x000000030F400700: "pvc_vg",
    0x0000000500400400: "bmg_g21",
    0x0000000500800000: "bmg_g31",
    0x0000000501000400: "lnl_m",
    0x0000000780000400: "ptl_h",
    0x0000000780400100: "ptl_u",
}


def device_isa() -> Optional[str]:
    """This process's GPU as its SYCL architecture name (e.g. "bmg_g21"), an
    "arch_0x..." name for one this table does not know, or None without an XPU."""
    if not torch.xpu.is_available():
        return None
    arch = torch.xpu.get_device_properties(torch.xpu.current_device()).architecture
    return _SYCL_ARCHITECTURES.get(arch, f"arch_{arch:#x}")


def deadline_s(watchdog_timeout: Optional[float]) -> float:
    """The deadline of every cross-GPU wait: half the watchdog, at most MAX_DEADLINE_S."""
    if not watchdog_timeout or watchdog_timeout <= 0:
        return MAX_DEADLINE_S
    return min(MAX_DEADLINE_S, 0.5 * float(watchdog_timeout))


def _ragged_verify(cfg: Any) -> bool:
    mode = envs.SGLANG_RAGGED_VERIFY_MODE.get().lower()
    return cfg.speculative_algorithm is not None and mode != "static"


# One probe a feature, on the resolved settings; a spec accepts features by listing
# them. PARTITIONED_GPU needs the loaded device, so the runtime checks it.
FEATURE_PROBES = {
    Feature.PIPELINE_PARALLEL: lambda c: c.pp_size > 1,
    Feature.MULTI_NODE_TP: lambda c: c.nnodes > 1,
    Feature.DP_ATTENTION: lambda c: (
        bool(c.enable_dp_attention) or (c.attn_dp_size or 1) > 1
    ),
    Feature.EXPERT_PARALLEL: lambda c: (c.ep_size or 1) > 1,
    Feature.EPLB: lambda c: bool(c.enable_eplb),
    Feature.CONTEXT_PARALLEL: lambda c: (
        (c.attn_cp_size or 1) > 1 or (c.dcp_size or 1) > 1 or bool(c.enable_prefill_cp)
    ),
    Feature.SPECULATIVE_DECODING: lambda c: c.speculative_algorithm is not None,
    Feature.RAGGED_VERIFY: _ragged_verify,
    Feature.TWO_BATCH_OVERLAP: lambda c: bool(
        c.enable_two_batch_overlap or c.enable_single_batch_overlap
    ),
    Feature.LORA: lambda c: bool(c.enable_lora or c.lora_paths),
    Feature.PD_DISAGGREGATION: lambda c: (c.disaggregation_mode or "null") != "null",
    Feature.RETURN_ROUTED_EXPERTS: lambda c: bool(c.enable_return_routed_experts),
    Feature.RETURN_INDEXER_TOPK: lambda c: bool(c.enable_return_indexer_topk),
    Feature.KV_OFFLOAD: lambda c: bool(
        c.enable_hierarchical_cache
        or c.disaggregation_decode_enable_offload_kvcache
        or c.enable_lmcache
        or c.enable_flexkv
    ),
    Feature.HISPARSE: lambda c: bool(c.enable_hisparse),
    Feature.MEMORY_RELEASE: lambda c: bool(
        c.enable_memory_saver or c.enable_weights_cpu_backup
    ),
    Feature.CPU_OFFLOAD: lambda c: (
        (c.cpu_offload_gb or 0) > 0 or (c.offload_group_size or 0) > 0
    ),
    Feature.TORCH_COMPILE: lambda c: bool(c.enable_torch_compile),
    Feature.DETERMINISTIC_INFERENCE: lambda c: bool(
        c.enable_deterministic_inference or c.rl_on_policy_target
    ),
}


def features_in_use(cfg: Any) -> set[Feature]:
    return {feature for feature, probe in FEATURE_PROBES.items() if probe(cfg)}


def plugin_reasons() -> list[str]:
    """Why the plugin that registers the specs will not load ([] when it will)."""
    if ENTRY_POINT not in {ep.name for ep in entry_points(group=GENERAL_PLUGINS_GROUP)}:
        return [
            f"the {ENTRY_POINT!r} plugin is not installed; install sgl-intel-giga "
            f"({PLUGIN_URL})"
        ]
    allowed = envs.SGLANG_PLUGINS.get()
    if allowed and ENTRY_POINT not in {x.strip() for x in allowed.split(",")}:
        return [f"SGLANG_PLUGINS={allowed!r} does not list the {ENTRY_POINT!r} plugin"]
    return []


def gigakernel_disable_reasons(
    *, cfg: Any, hf_config: Any, isa: Optional[str]
) -> tuple[Optional[GigakernelSpec], list[str]]:
    """The spec for this model and every reason it cannot serve this configuration."""
    if isa is None:
        return None, ["no XPU is visible"]
    spec, why = find_spec(architectures=hf_config.architectures, isa=isa)
    if spec is None and not registered_architectures():
        # the plugin loader logs and skips a plugin that fails, so nothing registered
        why = (
            f"the {ENTRY_POINT!r} plugin is installed but registered no spec; see "
            "'Failed to load plugin' or 'Failed to execute general plugin' above"
        )
    if spec is None:
        return None, [why]
    req = spec.requires
    reasons = [
        f"{f.value} is in use and {spec.name} does not accept it"
        for f in sorted(features_in_use(cfg) - spec.features, key=lambda f: f.name)
    ]
    one_of = [
        ("tp_size", cfg.tp_size, req.tp_sizes),
        (
            "decode graphs",
            cfg.cuda_graph_config.decode.backend,
            req.decode_graph_backends,
        ),
    ] + [(arg, getattr(cfg, arg), ok) for arg, ok in req.server_args.items()]
    exactly = [
        # another model's config may not have the field at all
        (f"config.{key}", getattr(hf_config, key, None), want)
        for key, want in req.hf_config.items()
    ] + [(name, getattr(envs, name).get(), want) for name, want in req.env.items()]
    reasons += [
        f"{what} {value!r} is not one of {sorted(map(str, allowed))}"
        for what, value, allowed in one_of
        if allowed and value not in allowed
    ]
    reasons += [
        f"{what} {value!r} is not {want!r}"
        for what, value, want in exactly
        if value != want
    ]
    watchdog = cfg.watchdog_timeout
    if watchdog and deadline_s(watchdog) < MIN_DEADLINE_S:
        reasons.append(
            f"--watchdog-timeout {watchdog:g} s leaves the waits a deadline under "
            f"{MIN_DEADLINE_S:g} s"
        )
    if spec.config_disable_reasons is not None:
        reasons += spec.config_disable_reasons(cfg, hf_config)
    return spec, reasons


def _dtype_reasons(spec: Optional[GigakernelSpec], model_config: Any) -> list[str]:
    # the resolved dtype, which the HF config alone does not give
    dtype = str(model_config.dtype).replace("torch.", "")
    if spec is None or not spec.requires.dtypes or dtype in spec.requires.dtypes:
        return []
    return [f"dtype {dtype!r} is not one of {sorted(spec.requires.dtypes)}"]


def handle_intel_gigakernel(server_args: Any) -> None:
    """Resolution step: fail before the weights load when the flag cannot be served,
    listing every reason."""
    cfg = resolving_view(server_args)
    if not cfg.enable_intel_gigakernel:
        return
    reasons = plugin_reasons()
    if not reasons:
        model_config = model_config_of(server_args)
        spec, reasons = gigakernel_disable_reasons(
            cfg=cfg, hf_config=model_config.hf_config, isa=device_isa()
        )
        reasons += _dtype_reasons(spec, model_config)
    # each replaces the model's forward, which would fail only after weights load
    reasons += [
        f"{flag} is also set; use one of the two"
        for flag, enabled in (
            ("--enable-nvidia-megakernel", cfg.enable_nvidia_megakernel),
            ("--enable-amd-monokernel", cfg.enable_amd_monokernel),
        )
        if enabled
    ]
    if reasons:
        raise ValueError(
            f"{FLAG} cannot run this setup:\n  - " + "\n  - ".join(reasons)
        )
    logger.warning("%s is experimental. Its kernels and coverage may change.", FLAG)
    logger.info("%s: %s on %s", FLAG, spec.name, device_isa())


def gigakernel_enabled() -> bool:
    """Whether this server runs with --enable-intel-gigakernel."""
    # the model loader also runs without a server's config (tools, unit tests)
    if not get_context().is_config_namespace_published("exec"):
        return False
    return get_exec().kernel.enable_intel_gigakernel


def install_gigakernel(
    *,
    model: torch.nn.Module,
    hf_config: PretrainedConfig,
    capture_tail_hooks: list[Callable[..., None]],
    is_draft: bool,
) -> None:
    """Build the model's spec and install its forward; runs before KV cache sizing."""
    if not gigakernel_enabled():
        return
    spec = _find_spec_on_every_rank(hf_config=hf_config, is_draft=is_draft)
    if spec is None:
        return
    # Resolution checked the target's spec on the server arguments; a draft's spec
    # is checked only here, on what the runner reads (no server-argument checks).
    reasons = _requirement_reasons(spec=spec, hf_config=hf_config)
    reasons += _model_disable_reasons(spec=spec, model=model, hf_config=hf_config)
    _refuse(_gather_reasons(reasons))

    tokens = _tokens_per_request()
    binding = GigakernelBinding(model=model)
    context = BuildContext(
        widths=tuple(range(tokens, spec.max_rows + 1, tokens)),
        tokens_per_request=tokens,
        deadline_s=deadline_s(get_device().watchdog_timeout),
        binding=binding,
    )
    start, free_before = time.perf_counter(), torch.xpu.mem_get_info()[0]
    built = spec.build(model, hf_config, context)
    # not setattr: a built object that is an nn.Module must not join the state dict
    model.__dict__["_gigakernel"] = built
    # Different widths would leave ranks waiting in different collectives.
    gigakernel_agree(tuple(built.widths), what=f"widths {spec.name} built")
    if not built.widths:
        _refuse([f"{spec.name} built no width of {list(context.widths)}"])
    # Captured kernels keep state derived from the weights that updates cannot refresh.
    model._derived_weight_cache_error = (
        f"Online weight updates are not supported with {FLAG} ({spec.name})"
    )
    logger.info(
        "Intel gigakernel: active (%s: %s; widths %s; %s numerics; built in %.1f s, "
        "%.1f MiB; wait deadline %.0f s)",
        spec.name,
        built.summary(),
        list(built.widths),
        spec.numerics.kind.value,
        time.perf_counter() - start,
        (free_before - torch.xpu.mem_get_info()[0]) / 2**20,
        context.deadline_s,
    )
    dispatch = _Dispatch(built=built, original=model.forward, binding=binding)
    model.forward = dispatch.wrapped()
    if _decode_graphs_enabled():
        capture_tail_hooks.append(_CaptureCheck(spec=spec, dispatch=dispatch))


def built_gigakernel(model: torch.nn.Module) -> Optional[BuiltGigakernel]:
    """The object the spec built for this model, or None."""
    return model.__dict__.get("_gigakernel")


def gigakernel_agree(value: Any, *, what: str, group: Any = None) -> None:
    """Raise on every rank unless every rank of group (default: the TP group) holds
    the same value; ranks that differ would wait in different collectives."""
    values = _gather_tp(value, group=group)
    if any(v != value for v in values):
        raise RuntimeError(
            f"{FLAG}: the ranks' {what} differ: "
            + "; ".join(f"rank {rank}: {v}" for rank, v in enumerate(values))
        )


def _find_spec_on_every_rank(
    *, hf_config: PretrainedConfig, is_draft: bool
) -> Optional[GigakernelSpec]:
    """The spec every rank found; for a draft model, None when no rank has one
    registered; else raise on every rank."""
    archs = hf_config.architectures
    spec, why = find_spec(architectures=archs, isa=device_isa())
    # Ranks branch together; one skipping the collectives below would hang its peers.
    lookups = _gather_tp(
        (
            spec_registered(archs),
            why,
            None if spec is None else (spec.name, spec.api_version),
        )
    )
    if not any(is_registered for is_registered, _, _ in lookups):
        if is_draft:
            # A draft model may have no registered spec.
            logger.info("Intel gigakernel: %s; %s keeps its own forward", why, archs)
            return None
        # resolution found the target's spec, so the plugin failed to load here
        _refuse(
            [
                f"{why} in the model workers, though argument resolution found it; "
                "see 'Failed to load plugin' or 'Failed to execute general plugin' "
                "above"
            ]
        )
    _refuse(
        [
            f"rank {rank}: {why}"
            for rank, (_, why, identity) in enumerate(lookups)
            if identity is None
        ]
    )
    if len({identity for *_, identity in lookups}) > 1:
        raise ValueError(
            f"{FLAG}: the ranks load different specs: "
            + "; ".join(f"rank {r}: {ident}" for r, (*_, ident) in enumerate(lookups))
        )
    return spec


def _requirement_reasons(
    *, spec: GigakernelSpec, hf_config: PretrainedConfig
) -> list[str]:
    """The spec's requirements this runner can check without the server arguments."""
    req, tp_size = spec.requires, get_parallel().tp_size
    reasons = []
    # an empty set accepts any TP size, as at argument resolution
    if req.tp_sizes and tp_size not in req.tp_sizes:
        reasons.append(f"tp_size {tp_size} is not one of {sorted(req.tp_sizes)}")
    graph = get_exec().graph.cuda_graph_config
    backend = Backend.DISABLED if graph is None else graph.decode.backend
    if req.decode_graph_backends and backend not in req.decode_graph_backends:
        reasons.append(
            f"decode graphs {backend!r} are not one of "
            f"{sorted(req.decode_graph_backends)}"
        )
    exactly = [
        (f"config.{key}", getattr(hf_config, key, None), want)
        for key, want in req.hf_config.items()
    ] + [(name, getattr(envs, name).get(), want) for name, want in req.env.items()]
    reasons += [
        f"{what} {v!r} is not {want!r}" for what, v, want in exactly if v != want
    ]
    return reasons


def _model_disable_reasons(
    *, spec: GigakernelSpec, model: torch.nn.Module, hf_config: PretrainedConfig
) -> list[str]:
    """This rank's reasons the loaded model cannot run the spec."""
    reasons = []
    if spec.model_disable_reasons is not None:
        reasons += spec.model_disable_reasons(model, hf_config)
    if "forward" in model.__dict__:
        reasons.append("the model's forward is already replaced on this instance")
    cores = torch.xpu.get_device_properties(
        torch.xpu.current_device()
    ).gpu_subslice_count
    if cores < spec.requires.resident_blocks:
        reasons.append(
            f"{Feature.PARTITIONED_GPU.value}: {cores} Xe cores, the persistent grids "
            f"need {spec.requires.resident_blocks}"
        )
    return reasons


class _Dispatch:
    """The installed forward: covered steps run the built gigakernel."""

    def __init__(
        self, *, built: BuiltGigakernel, original: Callable, binding: GigakernelBinding
    ):
        self.built, self.original, self.binding = built, original, binding
        # batch size -> the step planned while capturing its decode graphs; plan reads
        # only the mode and rows, so every graph variant of a batch size plans alike
        self.captured: dict[int, Optional[Step]] = {}

    def wrapped(self) -> Callable:
        # Graph runners and supports_pp inspect the original forward's signature.
        @functools.wraps(self.original)
        def forward(input_ids, positions, forward_batch, *args, **kwargs):
            step = self.built.plan(input_ids, forward_batch, *args, **kwargs)
            if torch.xpu.is_current_stream_capturing():
                self.captured[int(forward_batch.batch_size)] = step
            if step is None:
                return self.original(
                    input_ids, positions, forward_batch, *args, **kwargs
                )
            return self.built.forward(
                msgspec.structs.replace(step, view=self.binding.forward_view()),
                self.original,
                input_ids,
                positions,
                forward_batch,
                *args,
                **kwargs,
            )

        return forward


class _CaptureCheck:
    """Once every decode graph is captured, log the ones that run the gigakernel and
    refuse a runner none of whose graphs do."""

    def __init__(self, *, spec: GigakernelSpec, dispatch: _Dispatch):
        self.spec, self.dispatch = spec, dispatch
        self.is_done = False

    def __call__(
        self,
        runner: DecodeCudaGraphRunner,
        out: Any,
        forward_batch: ForwardBatch,
        num_tokens: int,
    ) -> None:
        captured = self.dispatch.captured
        if self.is_done or not set(runner.capture_bs) <= set(captured):
            return
        self.is_done = True
        self.dispatch.binding.assert_live()
        routed = sorted(bs for bs, step in captured.items() if step is not None)
        if not routed:
            raise ValueError(
                f"{FLAG}: none of the {len(captured)} captured decode graphs "
                f"(batch sizes {sorted(captured)}) runs {self.spec.name}"
            )
        logger.info(
            "Intel gigakernel: %s runs the decode graphs of batch sizes %s (%d of "
            "%d); every other step runs the standard forward",
            self.spec.name,
            routed,
            len(routed),
            len(captured),
        )


def _tokens_per_request() -> int:
    """Tokens per request of this server's decode-phase steps."""
    spec_config = get_spec()
    if spec_config.speculative_algorithm is None:
        return 1
    return int(spec_config.speculative_num_draft_tokens or 1)


def _decode_graphs_enabled() -> bool:
    graph = get_exec().graph.cuda_graph_config
    return graph is not None and graph.decode.backend != Backend.DISABLED


def _gather_tp(obj: Any, group: Any = None) -> list[Any]:
    """Gather a value from every rank of group (default: TP) through its CPU group."""
    group = get_parallel().tp_group if group is None else group
    if group.world_size == 1:
        return [obj]
    out: list[Any] = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def _gather_reasons(reasons: list[str]) -> list[str]:
    return [
        f"rank {rank}: {reason}"
        for rank, rank_reasons in enumerate(_gather_tp(reasons))
        for reason in rank_reasons
    ]


def _refuse(reasons: list[str]) -> None:
    if reasons:
        raise ValueError(
            f"{FLAG} cannot run this setup:\n  - " + "\n  - ".join(reasons)
        )

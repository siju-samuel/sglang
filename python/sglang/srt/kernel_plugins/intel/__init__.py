"""Intel gigakernels: whole-step dispatch for out-of-tree gigakernel plugins.

Plugins register specs through the sglang.srt.plugins entry point. With
--enable-intel-gigakernel the runtime validates the configuration, builds the
spec after the model is prepared, and routes covered steps through it; other
steps use the model forward.
"""

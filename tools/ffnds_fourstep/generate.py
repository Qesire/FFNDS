"""Active four-step FFNDS inference. generate_reference.py is the frozen school runner."""
import argparse
import gc
from time import perf_counter

import torch

from fourstep_backend import DEFAULT_VARIANT, PRIMARY_VARIANTS, make_port
from generate_reference import Experiment as ReferenceExperiment
from generate_reference import digest, equal_trace, load, save


_DEFAULT_PORT = object()


class Experiment(ReferenceExperiment):
    def __init__(self, variant=DEFAULT_VARIANT):
        if variant not in PRIMARY_VARIANTS:
            raise ValueError(f"Unknown primary four-step variant {variant!r}")
        self.variant = variant
        super().__init__()

    def build(self, mode=None):
        if mode is None:
            mode = self.variant
        gc.collect()
        torch.cuda.empty_cache()
        start = perf_counter()
        port = make_port(self.base, mode)
        torch.cuda.synchronize()
        self.state["setup"].append(dict(mode=mode, seconds=perf_counter() - start))
        return port

    def render(self, source, group, method=_DEFAULT_PORT, reuse=False):
        """Reuse one default FFNDS port across clips. Pass None for native Dense."""
        if method is _DEFAULT_PORT:
            if not hasattr(self, "_default_port"):
                self._default_port = self.build()
            method = self._default_port
        if reuse:
            # A video in this group may have been generated with another method.
            receipt = self.output / group / (source + ".json")
            if receipt.is_file():
                previous = load(receipt)
                variant = getattr(method, "variant", None)
                if method is None:
                    reuse = "counters" not in previous
                elif variant is None:
                    # Historical receipts do not uniquely identify these backends.
                    reuse = False
                else:
                    reuse = previous.get("counters", {}).get("variant") == variant
        return super().render(source, group, method, reuse=reuse)

    def run(self, source="vbi_0392", group=None):
        return self.render(source, self.variant if group is None else group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="vbi_0392")
    parser.add_argument("--variant", choices=PRIMARY_VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--group", help="Output group (defaults to the selected variant)")
    args = parser.parse_args()
    with torch.inference_mode():
        Experiment(variant=args.variant).run(args.source, args.group)

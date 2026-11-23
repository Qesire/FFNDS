"""Four-step Wan FFNDS backends. New ports use the validated wide8192 method.

Configuration is per resident port. The normal trajectory/bind API and native
forward path are retained. Use one port for serialized HHLL requests.
"""
from contextlib import contextmanager


DEFAULT_VARIANT = "wide8192"
JOINED_VARIANT = "wide8192_all_int8_joined"
PRIMARY_VARIANTS = (DEFAULT_VARIANT, JOINED_VARIANT)
_SPLIT_TILES = {
    "inplace512": 512, "wide8192": 8192, "wide12288": 12288,
    "wide16384": 16384, "wide31668": 31668,
}


def make_port(scope, variant=DEFAULT_VARIANT):
    """Construct a fresh port, with a single site pack per expert/layer."""
    if variant == JOINED_VARIANT:
        from y0_join import make_port as make_int8_port
        return _JoinedInt8Port(make_int8_port(scope))

    import exact_port

    if variant in ("fast512", "resident512", "original512"):
        return exact_port.make_port(scope, variant)
    names = {"anchor", "direct8192", "direct_inplace8192", *_SPLIT_TILES}
    if variant not in names:
        raise ValueError(f"Unknown FFNDS four-step variant {variant!r}")
    inner = exact_port.make_port(scope, "fast512")
    if variant in ("direct8192", "direct_inplace8192"):
        import fourstep_direct_hit as direct
        def pack(phase, params, device):
            prepared = direct.prepare(phase, params, device, mode="direct", tile_rows=8192)
            prepared["_fourstep_direct"]["spec"]["reuse_readout"] = variant == "direct_inplace8192"
            return prepared
        inner.prepare_static = pack
        inner.hit_prepared = direct.hit
    elif variant in _SPLIT_TILES:
        from fourstep_hit import make_hit
        inner.hit_prepared = make_hit(_SPLIT_TILES[variant], reuse_base=True)
    return _FourStepPort(inner, variant)


class _JoinedInt8Port:
    """Run the validated joined arm and identify it in saved trajectory receipts."""
    variant = JOINED_VARIANT

    def __init__(self, port):
        port.set_mode("joined")
        self._port = port
        self.inner = port.inner

    @property
    def counters(self):
        return dict(self._port.counters, variant=self.variant)

    @contextmanager
    def bind(self, owner, transformer_sha):
        with self._port.bind(owner, transformer_sha):
            yield self

    @contextmanager
    def trajectory(self):
        with self._port.trajectory():
            yield self


class _FourStepPort:
    def __init__(self, inner, variant):
        self.inner = inner
        self.variant = variant
        self.anchor_stats = {}

    @property
    def counters(self):
        stats = dict(self.inner.counters)
        tile_rows = _SPLIT_TILES.get(self.variant, 512 if self.variant == "anchor" else 8192)
        stats.update(variant=self.variant, tile_rows=tile_rows,
                     anchor_direct=self.anchor_stats.get("applied", 0),
                     anchor_fallback=self.anchor_stats.get("fallback", 0))
        return stats

    def bind(self, owner, transformer_sha):
        return self.inner.bind(owner, transformer_sha)

    @contextmanager
    def trajectory(self):
        from fourstep_anchor import bind_anchor
        with self.inner.trajectory():
            self.anchor_stats = {}
            with bind_anchor(self.inner, self.anchor_stats):
                yield self

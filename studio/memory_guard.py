"""16 GB Apple Silicon friendly memory guard policy for YuE2 Studio."""
from __future__ import annotations

import psutil

_GIB = 2**30
_MIB = 2**20
_PATCHED = False


def install_16gb_friendly_guard() -> bool:
    """Relax mlx-Yue's 32/128 GB benchmark memory guard on <=24 GB Macs.

    Upstream GPUExecution aborts immediately on macOS yellow memory pressure (level=2),
    <2 GiB available RAM, or 64 MiB of swap-out — which routinely triggers on 16 GB
    Apple M1 Macs even while macOS is comfortably compressing idle background apps.

    On <=24 GB Macs, we still protect the system from OOM/thrashing by aborting on:
      - macOS CRITICAL memory pressure (level >= 4)
      - Process physical footprint exceeding max(budget, 13.5 GiB)
      - Severe continuous swapping (> 3 GiB new swap-out in a single job)
    """
    global _PATCHED
    total_gib = psutil.virtual_memory().total / _GIB
    if total_gib >= 24.0 or _PATCHED:
        return False

    from lyra import measure

    def _studio_16gb_check_sample(self, sample):
        if self._baseline is None:
            self._baseline = sample
        footprint = sample.get("physical_footprint_bytes", 0)
        hard_cap_gib = max(self.memory_budget_gib, min(13.5, total_gib - 2.0))
        if footprint > hard_cap_gib * _GIB:
            raise MemoryError(
                f"Process footprint {footprint / _GIB:.2f} GiB exceeds "
                f"16 GB Mac safety ceiling ({hard_cap_gib:.1f} GiB)"
            )
        pressure = sample.get("system_memory_pressure_level", 1)
        # macOS levels: 1=NORMAL (green), 2=WARN (yellow, normal on 16GB), 4=CRITICAL (red)
        if pressure >= 4:
            raise MemoryError(
                f"System memory pressure is CRITICAL (level={pressure}). "
                "Close other apps to free RAM."
            )
        swapped = sample.get("system_swap_out_bytes", 0) - self._baseline.get("system_swap_out_bytes", 0)
        if swapped > 3072 * _MIB:
            raise MemoryError(
                f"Stopping GPU workload due to heavy swapping ({swapped / _MIB:.0f} MiB swapped out)"
            )

    measure.GPUExecution._check_sample = _studio_16gb_check_sample
    _PATCHED = True
    return True

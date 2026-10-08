"""Pay-per-event helper with graceful fallback for non-PPE runs (local, free, rental, pre-monetization).

Rules baked in:
- Charge only AFTER results are saved; never charge failed inputs.
- Use Actor.push_data(items, charged_event_name=EVENT) for per-item events: the SDK truncates the push to the
  user's max-total-charge budget and charges exactly what was stored.
- In non-PPE runs Actor.charge is a no-op (logs one warning); `budget()` returns None (= unlimited).
- Never charge synthetic `apify-*` events manually; remove `apify-default-dataset-item` in pricing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from apify import Actor


@dataclass
class Charger:
    primary_event: str
    counts: dict[str, int] = field(default_factory=dict)
    limit_reached: bool = False

    def is_ppe(self) -> bool:
        try:
            return bool(Actor.get_charging_manager().get_pricing_info().is_pay_per_event)
        except Exception:  # noqa: BLE001
            return False

    def budget(self, event: str | None = None) -> int | None:
        """How many more `event`s fit in the user's spending limit. None = unlimited / not PPE."""
        try:
            return Actor.get_charging_manager().calculate_max_event_charge_count_within_limit(event or self.primary_event)
        except Exception:  # noqa: BLE001
            return None

    async def push_and_charge(self, items: list[dict], event: str | None = None) -> int:
        """Push items and charge one `event` per stored item. Returns how many items were stored."""
        if not items or self.limit_reached:
            return 0
        event = event or self.primary_event
        b = self.budget(event)
        if b is not None and b < len(items):
            items = items[:b]
            self.limit_reached = True
        if not items:
            return 0
        try:
            res = await Actor.push_data(items, charged_event_name=event)
            if self.is_ppe() and res is not None:
                if res.charged_count < len(items):  # SDK stored only what fits in the budget
                    self.limit_reached = True
                    items = items[:res.charged_count]
                if res.event_charge_limit_reached:
                    self.limit_reached = True
        except Exception as e:  # noqa: BLE001 - never lose data because of a charging hiccup
            Actor.log.warning(f"Charging via push_data failed ({e}); pushing without charge.")
            await Actor.push_data(items)
        self.counts[event] = self.counts.get(event, 0) + len(items)
        return len(items)

    async def charge(self, event: str, count: int = 1) -> None:
        """Charge a non-item event (e.g. per document) after the work is saved."""
        if count <= 0:
            return
        try:
            res = await Actor.charge(event, count=count)
            if res.event_charge_limit_reached:
                self.limit_reached = True
        except Exception as e:  # noqa: BLE001
            Actor.log.warning(f"Charge {event} x{count} failed: {e}")
        self.counts[event] = self.counts.get(event, 0) + count

    async def push_free(self, items: list[dict] | dict) -> None:
        """Push items that are NOT billed (error rows, summaries)."""
        await Actor.push_data(items)

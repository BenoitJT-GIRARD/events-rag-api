"""Paging through the OpenDataSoft API that publishes OpenAgenda's events.

Upstream is a rolling window, so a fetch today and a fetch tomorrow return different
events. That is why the corpus is committed rather than rebuilt: every figure in the
README is about the frozen one.
"""

from datetime import datetime, timedelta

import httpx

from events_rag.config import get_settings
from events_rag.logger import get_logger

logger = get_logger(__name__)

BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/records"
)


class OpenAgendaClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.location_field = self.settings.location_field
        self.location_value = self.settings.location_value
        self.lang = self.settings.lang
        self.timezone = self.settings.timezone
        self.date_window_mode = self.settings.date_window_mode
        self.date_window_days = self.settings.date_window_days

    def _date_range(self) -> tuple[str, str | None]:
        today = datetime.now().date()

        if self.date_window_mode == "past":
            start = today - timedelta(days=self.date_window_days)
            end = today
        elif self.date_window_mode == "future":
            start = today
            end = today + timedelta(days=self.date_window_days)
        elif self.date_window_mode == "rolling":
            start = today - timedelta(days=self.date_window_days)
            end = None
        else:
            raise ValueError("date_window_mode must be one of: past, future, rolling")

        return start.isoformat(), end.isoformat() if end else None

    def _build_location_clause(self) -> str:
        value = self.location_value.replace("'", "\\'")

        field_map = {
            "city": "location_city",
            "region": "location_region",
            "department": "location_department",
        }

        if self.location_field not in field_map:
            raise ValueError("location_field must be one of: city, region, department")

        return f"{field_map[self.location_field]} = '{value}'"

    def _build_where_clause(self) -> str:
        start, end = self._date_range()
        location_clause = self._build_location_clause()

        date_clause = f"firstdate_begin >= date'{start}'"
        if end is not None:
            date_clause += f" AND firstdate_begin <= date'{end}'"

        return f"{location_clause} AND {date_clause}"

    async def fetch_events(self, limit: int = 100, offset: int = 0) -> list[dict]:
        params = {
            "limit": limit,
            "offset": offset,
            "where": self._build_where_clause(),
            "lang": self.lang,
            "timezone": self.timezone,
        }

        logger.info("opendatasoft.fetch_events", params=params)

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status()

        payload = response.json()
        events = payload.get("results", [])

        logger.info("opendatasoft.events_fetched", count=len(events))
        return events

    async def fetch_all_events(
        self,
        batch_size: int = 100,
        max_records: int = 5000,
    ) -> list[dict]:
        all_events: list[dict] = []
        offset = 0

        while offset < max_records:
            batch = await self.fetch_events(limit=batch_size, offset=offset)
            if not batch:
                break

            all_events.extend(batch)
            offset += batch_size

            if len(batch) < batch_size:
                break

        logger.info("opendatasoft.all_events_fetched", count=len(all_events))
        return all_events

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
        self.city = self.settings.city
        self.lang = self.settings.lang
        self.timezone = self.settings.timezone

    @staticmethod
    def _date_range() -> tuple[str, str]:
        start = datetime.now()
        end = start + timedelta(days=365)
        return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")

    def _build_where_clause(self) -> str:
        start, end = self._date_range()
        city = self.city.replace("'", "\\'")
        return (
            f"location_city = '{city}' "
            f"AND firstdate_begin >= date'{start[:10]}' "
            f"AND firstdate_begin <= date'{end[:10]}'"
        )

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

    async def fetch_all_events(self, batch_size: int = 100, max_records: int = 1000) -> list[dict]:
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
from events_rag.ingestion.openagenda_client import OpenAgendaClient


def test_build_where_clause_contains_location_and_date_filters() -> None:
    client = OpenAgendaClient()
    where_clause = client._build_where_clause()

    assert "firstdate_begin >=" in where_clause
    assert "firstdate_begin <=" in where_clause
    assert "location_" in where_clause
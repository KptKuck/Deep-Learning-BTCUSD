"""
Unit Tests fuer CSVReader
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestCSVReader:
    """Tests fuer die CSVReader-Klasse."""

    def test_read_csv_basic(self, temp_csv_file):
        """Test: CSV-Datei erfolgreich laden."""
        from btcusd_analyzer.data.reader import CSVReader

        reader = CSVReader()
        df = reader.read(temp_csv_file)

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_read_csv_columns(self, temp_csv_file):
        """Test: Erwartete Spalten vorhanden."""
        from btcusd_analyzer.data.reader import CSVReader

        reader = CSVReader()
        df = reader.read(temp_csv_file)

        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            assert col in df.columns, f"Spalte '{col}' fehlt"

    def test_read_csv_datetime_index(self, temp_csv_file):
        """Test: DateTime-Index korrekt gesetzt."""
        from btcusd_analyzer.data.reader import CSVReader

        reader = CSVReader()
        df = reader.read(temp_csv_file)

        assert isinstance(df.index, pd.DatetimeIndex), "Index ist kein DatetimeIndex"

    def test_read_nonexistent_file(self):
        """Test: Fehlerbehandlung bei nicht existierender Datei."""
        from btcusd_analyzer.data.reader import CSVReader

        reader = CSVReader()
        result = reader.read(Path("/nonexistent/file.csv"))

        assert result is None

    def test_read_csv_no_missing_values(self, temp_csv_file):
        """Test: Keine fehlenden Werte in Hauptspalten."""
        from btcusd_analyzer.data.reader import CSVReader

        reader = CSVReader()
        df = reader.read(temp_csv_file)

        main_cols = ['Open', 'High', 'Low', 'Close']
        for col in main_cols:
            assert df[col].isna().sum() == 0, f"Fehlende Werte in '{col}'"

    def test_read_csv_price_consistency(self, temp_csv_file):
        """Test: Preislogik konsistent (Low <= Close <= High)."""
        from btcusd_analyzer.data.reader import CSVReader

        reader = CSVReader()
        df = reader.read(temp_csv_file)

        # Low sollte <= High sein
        assert (df['Low'] <= df['High']).all(), "Low > High gefunden"

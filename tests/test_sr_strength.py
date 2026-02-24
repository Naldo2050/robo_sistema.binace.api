import pytest
import pandas as pd
from support_resistance.sr_strength import SRStrengthScorer


class TestSRStrengthScorer:
    """Tests for S/R Strength Scoring functionality."""

    def test_initialization(self):
        """Test if scorer initializes correctly."""
        scorer = SRStrengthScorer()
        assert scorer is not None
        assert hasattr(scorer, "score_levels")
        assert hasattr(scorer, "quick_score")

    def test_quick_score_basic(self):
        """Test quick score function."""
        scorer = SRStrengthScorer()
        
        score = scorer.quick_score(
            level_price=64888,
            current_price=64892,
            source="poc_daily",
            confluence_count=2
        )
        
        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_score_levels_no_candidates(self):
        """Test score levels with no candidates."""
        scorer = SRStrengthScorer()
        
        result = scorer.score_levels(
            current_price=64892,
            vp_data=None,
            pivot_data=None,
            ema_values=None,
            recent_candles=None
        )
        
        assert result["status"] == "no_candidates"
        assert len(result["levels"]) == 0
        assert len(result["supports"]) == 0
        assert len(result["resistances"]) == 0

    def test_score_levels_invalid_price(self):
        """Test score levels with invalid current price."""
        scorer = SRStrengthScorer()
        
        result = scorer.score_levels(
            current_price=-1,
            vp_data={"poc": 64888},
            pivot_data=None,
            ema_values=None,
            recent_candles=None
        )
        
        assert result["status"] == "invalid_price"
        assert len(result["levels"]) == 0

    def test_score_levels_with_basic_data(self):
        """Test score levels with basic data sources."""
        scorer = SRStrengthScorer()
        
        # Criar dados de teste básicos
        vp_data = {"poc": 64888, "vah": 66055, "val": 64683, "hvns": [65500, 64200]}
        pivot_data = {"classic": {"PP": 64850, "R1": 65100, "S1": 64600}}
        ema_values = {"ema_21_1h": 65418, "ema_21_4h": 66695}
        
        # Criar candles mock
        recent_candles = pd.DataFrame({
            "high": [64890, 64895, 64885, 64892, 64888],
            "low": [64880, 64885, 64875, 64880, 64878],
            "close": [64885, 64890, 64880, 64885, 64883]
        })
        
        result = scorer.score_levels(
            current_price=64892,
            vp_data=vp_data,
            pivot_data=pivot_data,
            ema_values=ema_values,
            recent_candles=recent_candles
        )
        
        assert result["status"] == "success"
        assert len(result["levels"]) > 0
        assert len(result["supports"]) > 0
        assert len(result["resistances"]) > 0
        assert result["nearest_support"] is not None
        assert result["nearest_resistance"] is not None
        
        # Verificar que todos os scores estão entre 0-100
        for level in result["levels"]:
            assert 0 <= level["strength"] <= 100
            assert "type" in level
            assert "price" in level
            assert "strength" in level

    def test_collect_candidates_from_multiple_sources(self):
        """Test if candidates are collected from multiple sources."""
        scorer = SRStrengthScorer()
        
        vp_data = {"poc": 64888, "vah": 66055, "val": 64683}
        pivot_data = {"classic": {"PP": 64850, "R1": 65100}}
        ema_values = {"ema_21_1d": 66000}
        
        # Acessar método interno (para teste)
        candidates = scorer._collect_candidates(
            current_price=64892,
            vp_data=vp_data,
            pivot_data=pivot_data,
            ema_values=ema_values,
            weekly_vp={"poc": 64900},
            monthly_vp={"poc": 65000}
        )
        
        assert len(candidates) > 0
        sources = [c["source"] for c in candidates]
        
        # Verificar se todas as fontes são coletadas
        assert "poc_daily" in sources
        assert "vah_daily" in sources
        assert "val_daily" in sources
        assert "poc_weekly" in sources
        assert "poc_monthly" in sources
        assert "pivot_classic_PP" in sources
        assert "pivot_classic_R1" in sources
        assert "ema_21_1d" in sources

    def test_merge_nearby_levels(self):
        """Test if nearby levels are merged."""
        scorer = SRStrengthScorer(touch_tolerance_pct=0.1)
        
        candidates = [
            {"price": 64888, "source": "poc_daily", "source_weight": 1.5},
            {"price": 64890, "source": "pivot_classic_PP", "source_weight": 1.3},
            {"price": 65100, "source": "vah_daily", "source_weight": 1.2},
            {"price": 65105, "source": "ema_21_1h", "source_weight": 0.8}
        ]
        
        merged = scorer._merge_nearby_levels(candidates, current_price=64892)
        
        assert len(merged) == 2
        for level in merged:
            assert "confluences" in level
            assert len(level["confluences"]) >= 1
            assert "confluence_count" in level
            assert level["confluence_count"] >= 1

    def test_count_touches_with_dataframe(self):
        """Test counting touches with pandas DataFrame."""
        scorer = SRStrengthScorer()
        
        # Criar candles com toques no nível 64888
        recent_candles = pd.DataFrame({
            "high": [64890, 64895, 64885, 64892, 64888],
            "low": [64880, 64885, 64875, 64880, 64878],
            "close": [64885, 64890, 64880, 64885, 64883]
        })
        
        touches = scorer._count_touches(64888, recent_candles)
        assert touches > 0

    def test_count_touches_with_list_of_dicts(self):
        """Test counting touches with list of dictionaries."""
        scorer = SRStrengthScorer()
        
        recent_candles = [
            {"high": 64890, "low": 64880, "close": 64885},
            {"high": 64895, "low": 64885, "close": 64890},
            {"high": 64885, "low": 64875, "close": 64880}
        ]
        
        touches = scorer._count_touches(64888, recent_candles)
        assert isinstance(touches, int)

    def test_round_number_candidates(self):
        """Test round number candidates generation."""
        scorer = SRStrengthScorer(round_number_interval=100)
        
        candidates = scorer._collect_candidates(
            current_price=64892,
            vp_data={"poc": 64888},  # Adicionar pelo menos uma fonte válida
            pivot_data=None,
            ema_values=None,
            weekly_vp=None,
            monthly_vp=None
        )
        
        # Deve gerar números redondos ao redor de 64892 (intervalo 100)
        round_prices = [c["price"] for c in candidates if c["source"] == "round_number"]
        
        assert len(round_prices) == 7  # 3 abaixo + 3 acima + base
        assert 64600 in round_prices
        assert 64700 in round_prices
        assert 64800 in round_prices
        assert 64900 in round_prices
        assert 65000 in round_prices
        assert 65100 in round_prices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
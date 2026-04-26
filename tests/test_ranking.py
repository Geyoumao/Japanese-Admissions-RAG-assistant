import unittest

from app.ranking import reciprocal_rank_fusion


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_fuses_two_rankings(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], constant=60)
        self.assertGreater(fused["b"], fused["a"])
        self.assertGreater(fused["c"], fused["d"])


if __name__ == "__main__":
    unittest.main()

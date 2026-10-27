import unittest
from unittest.mock import patch

from gtm_agent import gtm_agent as agent


class ProspectScoringTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "required_tech_stack": ["AWS", "Kafka", "dbt", "Snowflake"],
            "min_annual_revenue": 100,
            "description": "Analytics platform",
            "target_segment": "Enterprise",
        }
        self.profile = {"annual_revenue": 100, "tech_stack": ["AWS", "Kafka", "dbt"]}

    def score(self, profile=None, offering=None):
        return agent.score_prospect.invoke({
            "prospect_profile": self.profile if profile is None else profile,
            "offering": self.offering if offering is None else offering,
        })

    def test_coverage_and_justification_for_each_missing_technology(self):
        for missing in self.offering["required_tech_stack"]:
            with self.subTest(missing=missing):
                have = [tech for tech in self.offering["required_tech_stack"] if tech != missing]
                result = self.score({**self.profile, "tech_stack": have})
                self.assertEqual(result["rubric_breakdown"], {
                    "revenue_fit": 100, "tech_stack_match": 75, "component_max": 100,
                })
                self.assertEqual(result["score"], 88)
                self.assertIn(f"Required technologies present: {', '.join(have)}.", result["justification"])
                self.assertIn(f"Required technologies missing: {missing}.", result["justification"])
                self.assertNotIn("segment", str(result).lower())
                self.assertEqual(result, self.score({**self.profile, "tech_stack": have}))

    def test_revenue_and_weighted_score(self):
        for revenue, fit in [(0, 0), (25, 25), (73, 73), (100, 100), (200, 100)]:
            with self.subTest(revenue=revenue):
                result = self.score({**self.profile, "annual_revenue": revenue})
                self.assertEqual(result["rubric_breakdown"]["revenue_fit"], fit)
                self.assertEqual(result["score"], round(0.5 * fit + 0.5 * 75))

    def test_unscorable_offerings(self):
        for changes in [
            {"required_tech_stack": []}, {"required_tech_stack": None},
            {"min_annual_revenue": 0}, {"min_annual_revenue": None},
            {"min_annual_revenue": -1}, {"min_annual_revenue": "100"},
        ]:
            with self.subTest(changes=changes):
                offering = {**self.offering, **changes}
                result = self.score(offering=offering)
                self.assertIsNone(result["score"])
                self.assertIn("error", result)
                self.assertEqual(result, self.score(offering=offering))
        offering = dict(self.offering)
        del offering["min_annual_revenue"]
        self.assertIsNone(self.score(offering=offering)["score"])

    def test_missing_or_invalid_prospect_facts(self):
        for revenue in [None, -1, "100", float("nan"), float("inf"), True]:
            with self.subTest(revenue=revenue):
                self.assertIsNone(self.score({**self.profile, "annual_revenue": revenue})["score"])
        self.assertIsNone(self.score({"tech_stack": []})["score"])
        self.assertIsNone(self.score({"annual_revenue": 100})["score"])

    def test_empty_and_complete_coverage(self):
        for stack, expected in [([], 0), (self.offering["required_tech_stack"], 100)]:
            result = self.score({**self.profile, "tech_stack": stack})
            self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], expected)
            self.assertIn("missing: none." if expected else "present: none.", result["justification"])

    def test_saved_stack_overrides_stale_profile(self):
        offering = agent.lookup_offering.invoke({"offering_id": "OFFER-10002"})["offering"]
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-15229"})["prospect_profile"]
        required = offering["required_tech_stack"]
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=required[:3]):
            result = self.score({**profile, "tech_stack": required}, offering)
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 75)
        self.assertIn(f"Required technologies missing: {required[3]}.", result["justification"])


if __name__ == "__main__":
    unittest.main()

import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service
    from gtm_agent import gtm_agent as agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.records = patch.dict(data_service.PROSPECTS, deepcopy(data_service.PROSPECTS))
        self.profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        self.records.start()
        self.profiles.start()
        self.addCleanup(self.records.stop)
        self.addCleanup(self.profiles.stop)
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_added_technology_is_used_when_scoring_existing_profile(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-90001"})[
            "prospect_profile"
        ]
        offering = data_service.get_offering("OFFER-10007")
        update = agent.update_prospect_info.invoke(
            {"prospect_id": "LEAD-90001", "technology": "Okta"}
        )
        self.assertEqual(update, {
            "prospect_id": "LEAD-90001",
            "technology": "Okta",
            "updated": True,
            "found": True,
            "tech_stack": profile["tech_stack"] + ["Okta"],
        })
        self.assertEqual(data_service.fetch_tech_stack("LEAD-90001"), update["tech_stack"])

        def score(messages):
            payload = messages[1]["content"]
            supplied_profile = json.loads(payload.split("\n\nProspect profile:\n", 1)[1])
            self.assertEqual(supplied_profile["tech_stack"], update["tech_stack"])
            missing = set(offering["required_tech_stack"]) - set(supplied_profile["tech_stack"])
            self.assertEqual(missing, set())
            return agent.ProspectScore(
                score=100,
                justification="Has Snowflake, AWS, Okta, Kafka. Missing required technologies: none.",
                rubric_breakdown=agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100, segment_fit=100
                ),
            )

        with patch.object(agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score
            result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
        self.assertIn("Okta", result["justification"])
        self.assertEqual(result["score"], 100)

    def test_build_profile_after_update_includes_added_technology(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                data_service._PROFILES.clear()
                technology = "Okta" if not cached else "Redis"
                if cached:
                    agent.build_prospect_profile.invoke({"prospect_id": "LEAD-90001"})
                data_service.update_prospect_info("LEAD-90001", technology)
                result = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-90001"})
                self.assertTrue(result["found"])
                self.assertIn(technology, result["prospect_profile"]["tech_stack"])

    def test_duplicate_technology_preserves_success_and_stack(self):
        original = list(data_service.fetch_tech_stack("LEAD-90001"))
        result = data_service.update_prospect_info("LEAD-90001", original[0])
        self.assertEqual(result, {"updated": True, "found": True, "tech_stack": original})
        self.assertEqual(data_service.fetch_tech_stack("LEAD-90001"), original)

    def test_unknown_prospect_is_not_created(self):
        prospect_id = "LEAD-NONEXISTENT"
        result = data_service.update_prospect_info(prospect_id, "Okta")
        self.assertEqual(result, {"updated": False, "found": False})
        self.assertIsNone(data_service.get_prospect_record(prospect_id))
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": prospect_id}), {
            "prospect_profile": None, "found": False,
        })
        self.assertIsNone(data_service.get_profile_from_db(prospect_id)["prospect_profile"])


if __name__ == "__main__":
    unittest.main()

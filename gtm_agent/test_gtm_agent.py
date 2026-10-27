import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch.dict("os.environ", {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectToolsTest(unittest.TestCase):
    def setUp(self):
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.offering_id = next(iter(agent.data_service.OFFERINGS))
        self.rep = agent.REP_IDS[0]
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": self.rep["rep_id"]}})
        self.cache = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.scorer = Mock()
        self.scorer.invoke.return_value.model_dump.return_value = {"score": 75}
        scoring_patch = patch.object(agent, "_scoring_llm", self.scorer)
        scoring_patch.start()
        self.addCleanup(scoring_patch.stop)

    def test_score_with_ids_loads_records_and_caches_profile(self):
        result = agent.score_prospect.invoke({
            "prospect_id": self.prospect_id, "offering_id": self.offering_id,
        })
        self.assertEqual(result, {"score": 75})
        content = self.scorer.invoke.call_args.args[0][1]["content"]
        offering, profile = content.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering), agent.data_service.get_offering(self.offering_id))
        self.assertEqual(json.loads(profile)["prospect_id"], self.prospect_id)
        self.assertEqual(
            agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})["prospect_profile"],
            json.loads(profile),
        )

    def test_score_refreshes_cached_tech_stack(self):
        agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=["New technology"]):
            agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": self.offering_id})
        content = self.scorer.invoke.call_args.args[0][1]["content"]
        profile = json.loads(content.split("\n\nProspect profile:\n")[1])
        self.assertEqual(profile["tech_stack"], ["New technology"])

    def test_score_missing_records_and_invalid_offerings(self):
        for prospect_id, offering_id, error in [
            ("missing", self.offering_id, "Prospect not found."),
            (self.prospect_id, "missing", "Offering not found."),
        ]:
            with self.subTest(error=error):
                self.assertEqual(agent.score_prospect.invoke({
                    "prospect_id": prospect_id, "offering_id": offering_id,
                }), {"score": None, "error": error})
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            offering = dict(agent.data_service.get_offering(self.offering_id))
            offering.pop(field)
            with self.subTest(field=field), patch.object(agent.data_service, "get_offering", return_value=offering):
                self.assertEqual(agent.score_prospect.invoke({
                    "prospect_id": self.prospect_id, "offering_id": self.offering_id,
                }), {"score": None, "error": "Cannot score without a valid offering."})
        self.scorer.invoke.assert_not_called()

    def test_send_resolves_contacts_and_appends_signature(self):
        result = agent.send_prospect_email.func(self.prospect_id, "Hello", "Let's talk.", self.runtime)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], agent.data_service.get_prospect_record(self.prospect_id)["email"])
        self.assertEqual(result["from"], self.rep["email"])
        self.assertEqual(result["from_name"], self.rep["name"])
        self.assertEqual(result["subject"], "Hello")
        self.assertEqual(result["body"], "Let's talk.\n\n" + self.rep["name"] + "\n" + self.rep["email"])

    def test_send_missing_prospect_or_email(self):
        result = agent.send_prospect_email.func("missing", "Hello", "Body", self.runtime)
        self.assertEqual(result, {"status": "failed", "error": "Prospect not found."})
        with patch.object(agent.data_service, "get_prospect_record", return_value={"name": "Lead"}):
            result = agent.send_prospect_email.func(self.prospect_id, "Hello", "Body", self.runtime)
        self.assertEqual(result, {"status": "failed", "error": "Prospect record has no email address."})

    def test_send_requires_signed_in_rep_with_contact_details(self):
        for config in ({}, {"metadata": {}}, {"metadata": {"user_id": "missing"}}):
            with self.subTest(config=config):
                result = agent.send_prospect_email.func(
                    self.prospect_id, "Hello", "Body", SimpleNamespace(config=config),
                )
                self.assertEqual(result["status"], "failed")
                self.assertIn("rep", result["error"])
        for rep in ({"name": "Rep"}, {"email": "rep@example.com"}):
            with self.subTest(rep=rep), patch.object(agent.data_service, "get_rep", return_value=rep):
                result = agent.send_prospect_email.func(self.prospect_id, "Hello", "Body", self.runtime)
                self.assertEqual(result["status"], "failed")

    def test_rep_identity_lookup_remains_available(self):
        self.assertEqual(agent.get_current_rep.func(self.runtime), {"rep": self.rep, "found": True})
        self.assertEqual(agent.get_current_rep.func(SimpleNamespace(config={})), {"rep": None, "found": False})


if __name__ == "__main__":
    unittest.main()

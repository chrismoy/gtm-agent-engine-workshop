import copy
import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from deepagents import create_deep_agent

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")

SENSITIVE = {"billing_qualification", "tax_id", "date_of_birth", "card_on_file", "credit_check_ref"}
PROFILE_FIELDS = {"prospect_id", "name", "account_details", "annual_revenue", "tech_stack", "engagement_history"}
SCORE = agent.ProspectScore(
    score=65, justification="Some required technologies are missing.",
    rubric_breakdown=agent.RubricBreakdown(revenue_fit=80, tech_stack_match=50, segment_fit=65),
)


class RequestModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "privacy-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = {m.name: json.loads(m.content) for m in messages if isinstance(m, ToolMessage)}
        request = next(m.content for m in messages if m.type == "human")
        if "get_current_rep" not in results:
            name, args = "get_current_rep", {}
        elif request.startswith("Score"):
            if "build_prospect_profile" not in results:
                name, args = "build_prospect_profile", {"prospect_id": "LEAD-39002"}
            elif "lookup_offering" not in results:
                name, args = "lookup_offering", {"offering_id": "OFFER-10004"}
            elif "score_prospect" not in results:
                name, args = "score_prospect", {
                    "prospect_profile": results["build_prospect_profile"]["prospect_profile"],
                    "offering": results["lookup_offering"]["offering"],
                }
            else:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Scored."))])
        elif "get_prospect" not in results:
            name, args = "get_prospect", {"prospect_id": "LEAD-12853"}
        elif "send_prospect_email" not in results:
            name, args = "send_prospect_email", {
                "prospect": results["get_prospect"]["prospect"],
                "subject": "Demo next week", "body": "Can we schedule your demo next week?",
            }
        else:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Sent."))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="", tool_calls=[{"name": name, "args": args, "id": str(len(messages))}],
        ))])


class RunCapture(BaseCallbackHandler):
    def __init__(self):
        self.tool_results = []
        self.llm_inputs = []

    def on_tool_end(self, output, **kwargs):
        self.tool_results.append(output.content if isinstance(output, ToolMessage) else output)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.llm_inputs.extend(m.content for batch in messages for m in batch)


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cache = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def assert_private(self, value):
        serialized = json.dumps(value)
        for key in SENSITIVE:
            self.assertNotIn(key, serialized)

    def test_contact_allowlist_and_missing(self):
        record = {**agent.data_service.PROSPECTS["LEAD-12853"], "unrelated": "extra"}
        with patch.object(agent.data_service, "get_prospect_record", return_value=record):
            result = agent.get_prospect.invoke({"prospect_id": "LEAD-12853"})
        self.assertTrue(result["found"])
        self.assertEqual(result["prospect"], {
            "prospect_id": "LEAD-12853", **{k: record[k] for k in ("name", "email", "disqualified")},
        })
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {"prospect": None, "found": False})

    def test_new_and_legacy_cached_profiles(self):
        pid = "LEAD-39002"
        record = {**agent.data_service.PROSPECTS[pid], "unrelated": "extra"}
        with patch.object(agent.data_service, "get_prospect_record", return_value=record):
            profile = agent.build_prospect_profile.invoke({"prospect_id": pid})["prospect_profile"]
        self.assertEqual(set(profile), PROFILE_FIELDS)
        self.assertEqual(agent.data_service._PROFILES[pid], profile)
        for key in ("name", "annual_revenue", "account_details", "tech_stack", "engagement_history"):
            self.assertEqual(profile[key], agent.data_service.PROSPECTS[pid][key])
        agent.data_service._PROFILES[pid] = {**profile, "billing_qualification": {"tax_id": "private"}, "extra": 1}
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": pid})["prospect_profile"], profile)
        self.assertEqual(agent.data_service.get_profile_from_db(pid)["prospect_profile"], profile)
        self.assertEqual(agent.data_service._PROFILES[pid], profile)
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {"prospect_profile": None, "found": False})

    def test_scoring_sanitizes_nested_input_and_preserves_saved_stack(self):
        profile = {"prospect_id": "LEAD-39002", "tech_stack": ["fake"], **{k: "private" for k in SENSITIVE},
                   "account_details": {"items": [{"tax_id": "private", "segment": "Enterprise"}]}}
        original = copy.deepcopy(profile)
        llm = Mock()
        llm.invoke.return_value = SCORE
        with patch.object(agent, "_scoring_llm", llm):
            result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": agent.data_service.OFFERINGS["OFFER-10004"]})
        messages = llm.invoke.call_args.args[0]
        self.assert_private(messages)
        sent = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
        self.assertEqual(sent["tech_stack"], agent.data_service.fetch_tech_stack("LEAD-39002"))
        self.assertEqual(sent["account_details"], {"items": [{"segment": "Enterprise"}]})
        self.assertEqual(profile, original)
        self.assertEqual(result, SCORE.model_dump())

    def test_score_and_email_graph_runs(self):
        graph = create_deep_agent(model=RequestModel(), tools=[
            agent.get_current_rep, agent.get_prospect, agent.build_prospect_profile,
            agent.lookup_offering, agent.score_prospect, agent.send_prospect_email,
        ], system_prompt=agent.SYSTEM_PROMPT)
        for request, expected in (
            ("Score LEAD-39002 against OFFER-10004.", "score_prospect"),
            ("Can you email LEAD-12853 to schedule their demo call for next week?", "send_prospect_email"),
        ):
            with self.subTest(request=request):
                capture = RunCapture()
                llm = Mock()
                llm.invoke.return_value = SCORE
                original_invoke = graph.invoke

                def invoke_with_capture(inputs, config):
                    return original_invoke(inputs, {**config, "callbacks": [capture]})

                with patch.object(agent, "gtm_agent", graph), patch.object(agent, "_scoring_llm", llm):
                    with patch.object(graph, "invoke", side_effect=invoke_with_capture):
                        result = agent.run_agent(request, user_id=agent.REP_IDS[0]["rep_id"])
                self.assertIn(expected, [m.name for m in result["messages"] if isinstance(m, ToolMessage)])
                self.assertTrue(capture.tool_results)
                self.assertTrue(capture.llm_inputs)
                self.assert_private(capture.tool_results)
                self.assert_private(capture.llm_inputs)
                if llm.invoke.called:
                    self.assert_private(llm.invoke.call_args.args[0])


if __name__ == "__main__":
    unittest.main()

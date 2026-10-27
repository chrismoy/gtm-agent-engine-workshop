import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gtm_agent.gtm_agent import data_service, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.subject = "Book a demo"
        self.body = "Would you like to book a demo?"

    def send(self, prospect, **kwargs):
        return send_prospect_email.func(
            prospect, self.subject, self.body, self.runtime, **kwargs
        )

    def test_disqualified_source_blocks_even_if_contact_omits_or_denies_flag(self):
        for prospect_id in (
            "LEAD-50001", "LEAD-50002", "LEAD-50003", "LEAD-50004", "LEAD-50005"
        ):
            for fields in ({}, {"disqualified": False}):
                with self.subTest(prospect_id=prospect_id, fields=fields):
                    prospect = {
                        "prospect_id": prospect_id,
                        "email": "prospect@example.com",
                        **fields,
                    }
                    with patch("gtm_agent.gtm_agent.uuid.uuid4") as message_id:
                        result = self.send(prospect)
                    self.assertEqual(result, {
                        "status": "blocked",
                        "error": "Prospect is disqualified; outreach not permitted.",
                    })
                    message_id.assert_not_called()

    def test_missing_source_blocks(self):
        for fields in ({}, {"prospect_id": "missing"}):
            with self.subTest(fields=fields):
                with patch("gtm_agent.gtm_agent.uuid.uuid4") as message_id:
                    result = self.send({"email": "prospect@example.com", **fields})
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Prospect is disqualified; outreach not permitted.",
                })
                message_id.assert_not_called()

    def test_eligible_source_preserves_sent_message(self):
        prospect_id = "LEAD-70001"
        prospect = {
            "prospect_id": prospect_id,
            **data_service.get_prospect_record(prospect_id),
        }
        for from_rep in (None, {"name": "Sender", "email": "sender@example.com"}):
            with self.subTest(from_rep=from_rep):
                rep = from_rep or data_service.get_rep("rep_amills")
                with patch("gtm_agent.gtm_agent.uuid.uuid4") as message_id:
                    message_id.return_value.hex = "0123456789abcdef"
                    result = self.send(prospect, from_rep=from_rep)
                self.assertEqual(result, {
                    "status": "sent",
                    "message_id": "msg-0123456789ab",
                    "to": prospect["email"],
                    "to_name": prospect["name"],
                    "from": rep["email"],
                    "from_name": rep["name"],
                    "subject": self.subject,
                    "body": self.body,
                })

    def test_eligible_source_without_email_still_fails(self):
        result = self.send({"prospect_id": "LEAD-70001"})
        self.assertEqual(result, {
            "status": "failed",
            "error": "Prospect record has no email address.",
        })


if __name__ == "__main__":
    unittest.main()

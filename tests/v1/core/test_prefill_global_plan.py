# SPDX-License-Identifier: Apache-2.0
"""Pure plan contract; allocation and lifecycle are checked by integration tests."""
from types import SimpleNamespace
import unittest

from vllm.v1.core.sched.global_plan import NativePlan, OperationKind


class NativePlanContract(unittest.TestCase):
    def choose(self, fallback=None, known=False, submitted=True):
        request = SimpleNamespace(
            request_id="r", queue="waiting", accounted_tokens=0,
            status="WAITING", native_lookup_fallback=fallback,
            lookup_result_known=known, lookup_result_tokens=0 if known else None,
            lookup_submitted=submitted,
        )
        snapshot = SimpleNamespace(
            snapshot_id="step:1", requests=(request,),
            restore_preparation_enabled=True, new_lookup_enabled=True,
        )
        return NativePlan().plan(snapshot).attempts

    def test_native_failure_fallback_can_attempt_compute_without_fake_hit(self):
        for reason in ("ack_timeout", "status_timeout", "server_unhealthy",
                       "native_tracker_bypass"):
            with self.subTest(reason=reason):
                items = self.choose(fallback=reason)
                self.assertEqual([item.kind for item in items],
                                 [OperationKind.COMPUTE])

    def test_unknown_pending_query_is_not_a_local_compute_fallback(self):
        self.assertEqual(self.choose(), ())
        self.assertEqual(self.choose(known=True)[0].kind, OperationKind.COMPUTE)


if __name__ == "__main__":
    unittest.main()

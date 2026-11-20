import unittest

from checkpoint_validator import RecoveryView, recovery_view, validate


def make_version(parents, params, optimizer_tag, evals=None, exports=None,
                 timestamp=0):
    return {
        "parents": parents,
        "timestamp": timestamp,
        "params": params,
        "optimizer": optimizer_tag,
        "evals": evals or [],
        "exports": exports or [],
    }


def opt(obj_id, digest, version):
    return {obj_id: {"digest": digest, "param_version": version}}


class LinearHistoryTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "versions": {
                "v1": make_version([], {"p1": "d1"}, opt("o1", "od1", "v1"),
                                   timestamp=1),
                "v2": make_version(["v1"], {"p2": "d2"}, opt("o2", "od2", "v2"),
                                   evals=["e2"], timestamp=2),
                "v3": make_version(["v2"], {"p3": "d3"}, opt("o3", "od3", "v3"),
                                   exports=["x3"], timestamp=3),
            },
            "objects": {
                "p1": {"digest": "d1"}, "o1": {"digest": "od1"},
                "p2": {"digest": "d2"}, "o2": {"digest": "od2"},
                "e2": {"digest": "ed2"},
                "p3": {"digest": "d3"}, "o3": {"digest": "od3"},
                "x3": {"digest": "xd3"},
            },
            "deletions": [],
        }

    def test_all_versions_recoverable(self):
        report = validate(self.snapshot)
        self.assertEqual(report["recoverable"], ["v1", "v2", "v3"])
        self.assertEqual(report["unrecoverable"], {})
        self.assertEqual(report["missing_objects"], {})

    def test_ancestor_chain(self):
        view = recovery_view(self.snapshot, "v3")
        self.assertEqual(view.ancestors, ("v2", "v1"))

    def test_missing_parent_blocks_recovery(self):
        self.snapshot["versions"]["v2"]["parents"] = ["v0"]
        report = validate(self.snapshot)
        self.assertNotIn("v2", report["recoverable"])
        self.assertIn("missing_parent:v0", report["unrecoverable"]["v2"])

    def test_digest_mismatch_blocks_recovery(self):
        self.snapshot["objects"]["p2"]["digest"] = "tampered"
        report = validate(self.snapshot)
        self.assertNotIn("v2", report["recoverable"])
        self.assertIn("digest_mismatch:p2", report["unrecoverable"]["v2"])
        # v3 has its own full manifest and is unaffected.
        self.assertIn("v3", report["recoverable"])

    def test_optimizer_param_mismatch_blocks_recovery(self):
        self.snapshot["versions"]["v2"]["optimizer"]["o2"]["param_version"] = "v1"
        report = validate(self.snapshot)
        self.assertNotIn("v2", report["recoverable"])
        self.assertIn("optimizer_param_mismatch:o2",
                      report["unrecoverable"]["v2"])


class ForkedHistoryTest(unittest.TestCase):
    def test_fork_branches_recoverable_independently(self):
        snapshot = {
            "versions": {
                "base": make_version([], {"pb": "db"}, opt("ob", "odb", "base")),
                "fork-a": make_version(["base"], {"pa": "da"},
                                       opt("oa", "oda", "fork-a")),
                "fork-b": make_version(["base"], {"pb2": "db2"},
                                       opt("ob2", "odb2", "fork-b")),
            },
            "objects": {
                "pb": {"digest": "db"}, "ob": {"digest": "odb"},
                "pa": {"digest": "da"}, "oa": {"digest": "oda"},
                "pb2": {"digest": "db2"}, "ob2": {"digest": "odb2"},
            },
            "deletions": [],
        }
        report = validate(snapshot)
        self.assertEqual(report["recoverable"], ["base", "fork-a", "fork-b"])

        # Break one branch: the other branch and the base stay recoverable.
        del snapshot["objects"]["pa"]
        report = validate(snapshot)
        self.assertNotIn("fork-a", report["recoverable"])
        self.assertIn("fork-b", report["recoverable"])
        self.assertIn("base", report["recoverable"])
        self.assertEqual(report["missing_objects"]["fork-a"], ["pa"])


class SharedShardTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "versions": {
                "v1": make_version([], {"shared": "ds", "p1": "d1"},
                                   opt("o1", "od1", "v1"), timestamp=1),
                "v2": make_version(["v1"], {"shared": "ds", "p2": "d2"},
                                   opt("o2", "od2", "v2"), timestamp=2),
            },
            "objects": {
                "shared": {"digest": "ds"},
                "p1": {"digest": "d1"}, "o1": {"digest": "od1"},
                "p2": {"digest": "d2"}, "o2": {"digest": "od2"},
            },
            "deletions": [],
        }

    def test_shared_references_reported(self):
        report = validate(self.snapshot)
        self.assertEqual(report["shared_references"], {"shared": ["v1", "v2"]})
        view = recovery_view(self.snapshot, "v2")
        self.assertEqual(view.shared_objects, frozenset({"shared"}))

    def test_retirement_of_old_version_cannot_delete_shared_shard(self):
        # v1 is retired and cleanup tries to delete the shared shard.
        self.snapshot["deletions"].append(
            {"object": "shared", "retired_by": "v1", "at": 2})
        report = validate(self.snapshot)
        # Deletion is rejected: both versions remain recoverable.
        self.assertIn("shared", report["protected_deletions"])
        self.assertEqual(report["recoverable"], ["v1", "v2"])
        self.assertNotIn("shared", report["deleted_objects"])
        self.assertNotIn("shared", report["safe_cleanup"])


class OldVersionDeletionTest(unittest.TestCase):
    def test_deleted_shard_makes_old_version_unrecoverable(self):
        snapshot = {
            "versions": {
                "v1": make_version([], {"p1": "d1"}, opt("o1", "od1", "v1"),
                                   timestamp=1),
                "v2": make_version(["v1"], {"p2": "d2"}, opt("o2", "od2", "v2"),
                                   timestamp=2),
            },
            "objects": {
                "p1": {"digest": "d1"}, "o1": {"digest": "od1"},
                "p2": {"digest": "d2"}, "o2": {"digest": "od2"},
                "orphan": {"digest": "dz"},
            },
            # p1/o1 are only referenced by v1, which was retired.
            "deletions": [
                {"object": "p1", "retired_by": "v1", "at": 2},
                {"object": "o1", "retired_by": "v1", "at": 2},
            ],
        }
        report = validate(snapshot)
        self.assertEqual(report["recoverable"], ["v2"])
        self.assertEqual(report["missing_objects"]["v1"], ["o1", "p1"])
        self.assertIn("deleted_param:p1", report["unrecoverable"]["v1"])
        self.assertEqual(report["deleted_objects"], ["o1", "p1"])
        # Unreferenced and not deleted: safe to clean up.
        self.assertEqual(report["safe_cleanup"], ["orphan"])


class ManifestRollbackTest(unittest.TestCase):
    def test_rolled_back_manifest_referencing_deleted_shard(self):
        # v3's manifest was rolled back to an old manifest that still
        # references shard "p-old", which has since been deleted.
        snapshot = {
            "versions": {
                "v1": make_version([], {"p-old": "dold"},
                                   opt("o-old", "odold", "v1"), timestamp=1),
                "v2": make_version(["v1"], {"p2": "d2"}, opt("o2", "od2", "v2"),
                                   timestamp=2),
                "v3": make_version(["v2"], {"p-old": "dold"},
                                   opt("o3", "od3", "v3"), timestamp=3),
            },
            "objects": {
                "p2": {"digest": "d2"}, "o2": {"digest": "od2"},
                "o3": {"digest": "od3"},
            },
            "deletions": [
                {"object": "p-old", "retired_by": "v1", "at": 2},
                {"object": "o-old", "retired_by": "v1", "at": 2},
            ],
        }
        report = validate(snapshot)
        # v1 lost its shards, v3's rolled-back manifest dangles.
        self.assertEqual(report["recoverable"], ["v2"])
        self.assertEqual(report["missing_objects"]["v3"], ["p-old"])
        self.assertIn("deleted_param:p-old", report["unrecoverable"]["v3"])

    def test_rolled_back_manifest_with_stale_digest(self):
        snapshot = {
            "versions": {
                "v1": make_version([], {"p1": "d1-old"}, opt("o1", "od1", "v1"),
                                   timestamp=1),
            },
            "objects": {"p1": {"digest": "d1-new"}, "o1": {"digest": "od1"}},
            "deletions": [],
        }
        report = validate(snapshot)
        self.assertEqual(report["recoverable"], [])
        self.assertIn("digest_mismatch:p1", report["unrecoverable"]["v1"])


class ImmutableViewTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "versions": {
                "v1": make_version([], {"p1": "d1"}, opt("o1", "od1", "v1")),
            },
            "objects": {
                "p1": {"digest": "d1"},
                "o1": {"digest": "od1", "archived": True},
            },
            "deletions": [],
        }

    def test_view_is_immutable(self):
        view = recovery_view(self.snapshot, "v1")
        self.assertIsInstance(view, RecoveryView)
        with self.assertRaises(TypeError):
            view.objects["p1"] = "tampered"
        with self.assertRaises(AttributeError):
            view.shared_objects = frozenset()
        self.assertEqual(view.archived_objects, frozenset({"o1"}))

    def test_view_survives_later_cleanup(self):
        view = recovery_view(self.snapshot, "v1")
        # Later cleanup mutates the original snapshot.
        del self.snapshot["objects"]["p1"]
        self.snapshot["deletions"].append(
            {"object": "o1", "retired_by": "v2", "at": 9})
        # The previously produced view is unchanged.
        self.assertEqual(view.objects["p1"], "d1")
        self.assertEqual(view.objects["o1"], "od1")
        # A fresh validation reflects the cleanup.
        self.assertIsNone(recovery_view(self.snapshot, "v1"))

    def test_input_snapshot_not_mutated(self):
        before = repr(self.snapshot)
        validate(self.snapshot)
        self.assertEqual(repr(self.snapshot), before)


if __name__ == "__main__":
    unittest.main()

"""Checkpoint time-travel validator.

Validates whether any historical checkpoint version in a DAG can still be
fully restored, using *metadata only*. The validator never loads model
weights; it only inspects manifests, object digests, deletion records and
archive status.

Input snapshot schema (plain dict, JSON-serialisable)::

    {
        "versions": {
            "v1": {
                "parents": ["v0"],                 # parent version ids
                "timestamp": 1,                    # logical time (optional)
                "params": {"shard-a": "sha256:.."},# object id -> digest
                "optimizer": {                     # optimizer shards
                    "shard-opt": {"digest": "sha256:..", "param_version": "v1"}
                },
                "evals": ["eval-report-1"],        # referenced eval objects
                "exports": ["export-1"]            # exported model pointers
            }
        },
        "objects": {
            "shard-a": {"digest": "sha256:..", "archived": False}
        },
        "deletions": [
            {"object": "shard-old", "retired_by": "v2", "at": 2}
        ]
    }

A version is recoverable only when:
  * every ancestor version is present in the DAG (no missing parents);
  * every referenced param/optimizer/eval/export object exists, is not
    validly deleted, and its digest matches the manifest;
  * every optimizer shard is tagged with the version's own param version.

Deletion records are evaluated at their logical time ``at``: a deletion is
*rejected* (the object stays available) when a version that already existed
at that time, and is not itself retired, still references the object. Shared
shards referenced by newer live versions must not disappear because an old
time point was retired. A version created *after* the deletion (for example
through a manifest rollback) that references the deleted object simply finds
it missing.
"""

import copy
from types import MappingProxyType


class RecoveryView(object):
    """Immutable snapshot of a recoverable version.

    The view is built from a deep-copied, frozen state, so later cleanup or
    mutation of the input snapshot cannot silently change it.
    """

    __slots__ = ("_version", "_ancestors", "_objects", "_shared", "_archived")

    def __init__(self, version, ancestors, objects, shared, archived):
        self._version = version
        self._ancestors = tuple(ancestors)
        self._objects = MappingProxyType(dict(objects))
        self._shared = frozenset(shared)
        self._archived = frozenset(archived)

    @property
    def version(self):
        return self._version

    @property
    def ancestors(self):
        """Tuple of ancestor version ids, nearest parent first."""
        return self._ancestors

    @property
    def objects(self):
        """Read-only mapping of object id -> digest required for restore."""
        return self._objects

    @property
    def shared_objects(self):
        """Frozenset of objects also referenced by other versions."""
        return self._shared

    @property
    def archived_objects(self):
        """Frozenset of required objects that live in cold archive."""
        return self._archived

    def __repr__(self):
        return "RecoveryView(version=%r, objects=%d)" % (
            self._version, len(self._objects))


def _manifest_entries(manifest):
    """Yield (object_id, expected_digest, kind) for all required objects."""
    for obj_id, digest in (manifest.get("params") or {}).items():
        yield obj_id, digest, "param"
    for obj_id, entry in (manifest.get("optimizer") or {}).items():
        yield obj_id, entry.get("digest"), "optimizer"
    for obj_id in manifest.get("evals") or []:
        yield obj_id, None, "eval"
    for obj_id in manifest.get("exports") or []:
        yield obj_id, None, "export"


def _referenced_objects(manifest):
    return set(obj_id for obj_id, _, _ in _manifest_entries(manifest))


def _ancestors_of(version_id, versions):
    """Return (ancestors, missing_parent). Ancestors are nearest-first."""
    ancestors = []
    seen = {version_id}
    stack = list(versions[version_id].get("parents") or [])
    missing = None
    while stack:
        parent = stack.pop(0)
        if parent in seen:
            continue
        seen.add(parent)
        if parent not in versions:
            missing = parent
            continue
        ancestors.append(parent)
        stack.extend(versions[parent].get("parents") or [])
    return ancestors, missing


def validate(snapshot):
    """Validate a checkpoint snapshot and return a recovery report.

    The input is never mutated; all computation happens on a deep copy.
    """
    state = copy.deepcopy(snapshot or {})
    versions = state.get("versions") or {}
    objects = state.get("objects") or {}
    deletions = state.get("deletions") or []

    # Reference map: object id -> set of version ids that reference it.
    refs = {}
    for vid, manifest in versions.items():
        for obj_id in _referenced_objects(manifest):
            refs.setdefault(obj_id, set()).add(vid)

    # Classify deletion records against the versions that existed when the
    # deletion happened. A deletion is rejected (shared-shard protection)
    # when a live, non-retired version referenced the object at that time.
    retired = set(rec.get("retired_by") for rec in deletions)
    deleted = {}
    protected_deletions = {}
    for record in deletions:
        obj_id = record.get("object")
        at = record.get("at", 0)
        live_refs = [
            vid for vid in refs.get(obj_id, ())
            if versions[vid].get("timestamp", 0) <= at and vid not in retired
        ]
        if live_refs:
            protected_deletions[obj_id] = record
        else:
            deleted[obj_id] = record

    recoverable = []
    unrecoverable = {}
    missing_objects = {}
    views = {}

    for vid in sorted(versions):
        manifest = versions[vid]
        reasons = []
        missing = []

        ancestors, missing_parent = _ancestors_of(vid, versions)
        if missing_parent is not None:
            reasons.append("missing_parent:%s" % missing_parent)

        for obj_id, expected, kind in sorted(_manifest_entries(manifest)):
            if obj_id in deleted:
                missing.append(obj_id)
                reasons.append("deleted_%s:%s" % (kind, obj_id))
                continue
            record = objects.get(obj_id)
            if record is None:
                missing.append(obj_id)
                reasons.append("missing_%s:%s" % (kind, obj_id))
                continue
            if expected is not None and record.get("digest") != expected:
                reasons.append("digest_mismatch:%s" % obj_id)

        for obj_id, entry in sorted((manifest.get("optimizer") or {}).items()):
            if entry.get("param_version") != vid:
                reasons.append("optimizer_param_mismatch:%s" % obj_id)

        if reasons:
            unrecoverable[vid] = sorted(set(reasons))
            if missing:
                missing_objects[vid] = sorted(missing)
        else:
            recoverable.append(vid)
            required = {}
            archived = []
            for obj_id, expected, _ in _manifest_entries(manifest):
                required[obj_id] = objects[obj_id].get("digest")
                if objects[obj_id].get("archived"):
                    archived.append(obj_id)
            shared = [o for o in required if len(refs.get(o, ())) > 1]
            views[vid] = RecoveryView(vid, ancestors, required, shared, archived)

    shared_references = {
        obj_id: sorted(vids)
        for obj_id, vids in sorted(refs.items())
        if len(vids) > 1
    }

    # Safe cleanup: objects not yet deleted whose only referencing versions
    # (if any) are retired. Objects with rejected (protected) deletion
    # records are still referenced by live versions, so they never appear.
    safe_cleanup = sorted(
        obj_id
        for obj_id in objects
        if obj_id not in deleted and refs.get(obj_id, set()) <= retired
    )

    return {
        "recoverable": recoverable,
        "unrecoverable": unrecoverable,
        "missing_objects": missing_objects,
        "shared_references": shared_references,
        "protected_deletions": protected_deletions,
        "deleted_objects": sorted(deleted),
        "safe_cleanup": safe_cleanup,
        "views": views,
    }


def recovery_view(snapshot, version_id):
    """Return the immutable RecoveryView for one version, or None.

    The view is a point-in-time snapshot: mutating or cleaning up the input
    afterwards does not change it.
    """
    report = validate(snapshot)
    return report["views"].get(version_id)

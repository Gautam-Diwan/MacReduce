from macreduce.serving import cache_summary


def test_cache_summary_names_untrimmable_kinds():
    class ArraysCache:
        def is_trimmable(self):
            return False

    class KVCache:
        def is_trimmable(self):
            return True

    summary = cache_summary([KVCache(), ArraysCache(), KVCache()])
    assert summary["counts"] == {"KVCache": 2, "ArraysCache": 1}
    assert summary["untrimmable"] == ["ArraysCache"]
    assert summary["speculative_ok"] is False

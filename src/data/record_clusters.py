import re
from collections import Counter


def pack_record_clusters(rows, group_field, text_field, max_examples=3):
    """
    Group records by a field and build a compact summary for each bucket.

    Parameters
    ----------
    rows : list[dict]
        Input records.
    group_field : str
        Field used to define the group.
    text_field : str
        Field from which text snippets are extracted.
    max_examples : int
        Maximum number of example snippets stored per group.

    Returns
    -------
    list[dict]
        Each item contains:
        - group: the grouping value
        - count: number of rows in the group
        - signature: a short representative text fragment
        - keywords: the most frequent words from the group
        - examples: a few compact text samples
    """
    buckets = {}

    for row in rows:
        key = row.get(group_field)
        if key is None:
            continue

        bucket = buckets.setdefault(
            key,
            {
                "count": 0,
                "signature": "",
                "tokens": Counter(),
                "examples": [],
            },
        )

        bucket["count"] += 1

        text = str(row.get(text_field, "") or "").strip()
        if not text:
            continue

        if not bucket["signature"]:
            bucket["signature"] = re.sub(r"\s+", " ", text)[:120]

        words = re.findall(r"[A-Za-z0-9']+", text.lower())
        if words:
            bucket["tokens"].update(words[:12])

        if len(bucket["examples"]) < max_examples:
            compact = " ".join(words[:16]).strip()
            if compact:
                bucket["examples"].append(compact[:140])

    result = []
    for group, payload in buckets.items():
        top_terms = [term for term, _ in payload["tokens"].most_common(4)]
        result.append(
            {
                "group": group,
                "count": payload["count"],
                "signature": payload["signature"],
                "keywords": top_terms,
                "examples": payload["examples"],
            }
        )

    result.sort(key=lambda item: (-item["count"], str(item["group"])))
    return result

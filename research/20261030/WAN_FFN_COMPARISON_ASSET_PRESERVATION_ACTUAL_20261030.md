# Wan FFN comparison assets: actual preservation

The latest876-reference manifest resolves to874 original paths and564 distinct contents. All874 original paths were rehashed against their frozen SHA and length. Independent copies were then rehashed:562 content-addressed objects for the earlier868-reference manifest, plus8 path-preserving copies for the newly added H/X videos. Both operations actually exited0. Total independent copy payload is2594050464 bytes; original files were preserved without rename, overwrite or deletion.

The base operation read6,861,285,097 payload bytes with balanced predebits and completed counts, and took333.587s. Twenty legacy logical `HF:` prefixes were resolved within the HF root. The eight-video supplement used the newly merged non-overwriting copy tool, followed by a separate archive-root SHA verification:8/8 copied and8/8 reverified. Its payload-byte ledger is not instrumented. Exact base/delta manifest SHA, source commits, tool bindings, run/actual-exit observations and private vault locators are retained in the JSON.

Copies are in private same-host vaults outside repository and HF cache. Binary payloads are not uploaded to GitHub. This is file maintenance, not a model/scientific run, an offsite backup or protection from shared-storage failure. Historical report-only HF assets outside the exact manifest remain outside this copy scope.

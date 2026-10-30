# Wan FFN comparison assets: actual preservation

The newly merged868-reference manifest resolves to866 original paths and562 distinct contents. All866 original paths were rehashed against their frozen SHA and length. All562 independent-inode copies were rehashed after copying. The maintenance process actually exited0; payload read ledgers close at6,861,285,097 bytes, and independent copy payload is2,584,154,098 bytes. Runtime333.587 seconds, GPU0, one CPU. Three small integrity/path fixtures passed.

Twenty legacy references use the logical `HF:` prefix; it is removed only when resolving the original HF-root path. Original records and all original files are preserved without rename, overwrite or deletion. Content-addressed copies and the exact path-to-object index reside in a private vault outside both repository and HF cache; the private locator and SHA-bound run/exit observations are listed in the JSON. Binary payloads are not uploaded to GitHub.

These are same-host independent copies. This is file maintenance, not new scientific evaluation, an offsite backup or protection from shared-storage failure. Historical report-only assets outside the exact manifest remain outside this copy scope.

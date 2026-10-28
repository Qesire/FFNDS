# SM120 kernel overlay export v2 · independent review

**SUPPORT for exact version:** source SHA256 `c08b5baf1a4f6290a514a67d52bedf65dfc323202ddd5f3641a429d28ada6152`, tests `ddd26b5a2950bce6f6cac083df4e6a8819341a3a2e6ec410d111a31182a3565a`, draft `fd97e542591cad7c8cc16878dffa20cb078c592a614c0e63c9d43f4fa45fbe12`. This is SRC and synthetic CPU evidence; no export or GPU execution was performed.

The earlier readonly boundsource failure was caused by a changed access time in full stat comparison; no export scope had been frozen, registered or launched. V2 checks regular files and compares stable device, inode, size, mtime and ctime before/after reading, deliberately excluding access time. It independently checks all six content SHA identities, including ELF magic, with the five Python identities bound to the actual build protocol and the shared object bound to the actual build RUN and exit0. Archive entries have fixed order and deterministic metadata.

Two synthetic CPU controls actually passed. They exercise atime-only change, changed inode, same-size content mutation, symlink and non-ELF rejection, as well as deterministic output. Actual original-host boundsource, registered export, archive receipt and later separate native execution remain unmeasured by this review.

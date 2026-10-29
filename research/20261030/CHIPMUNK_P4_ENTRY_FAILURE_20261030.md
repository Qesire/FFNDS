# Chipmunk P4 entry failure

The first P4 launch exited 1 after 0.2333 seconds, before writing RUN or entering Torch, tensor or GPU work. A function-local `import importlib.util` made `importlib` local to the entry function; the earlier metadata lookup then raised `NameError`. This is a startup software failure and produces no scientific result. The failed scope remains closed and is not automatically retried.

The [whitelist receipt](CHIPMUNK_P4_ENTRY_FAILURE_20261030.json) binds the actual EXIT, protocol and job SHA. A separate V2 fixes the import and requires a separately registered isolated CPU entry preflight with its own actual successful RUN/EXIT before GPU execution.

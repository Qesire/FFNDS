# Author response to independent review V1

Initial reviewer disposition: REVISE. I agree that the two per-level sparse and dense correctness calls require predebit and durable completion accounting. Each correctness call now records predebit=1 and completion=UNKNOWN before dispatch; only after the call returns and its GPU stream synchronizes is completion=1 persisted. A failure or interrupted synchronization retains UNKNOWN, avoiding a claim that the call never occurred. Closed validation requires both completed calls at each of the two levels. A synthetic failure test checks the predebit and unknown state, and success checks completion. Six synthetic tests pass.

The author-style benchmark timing loops and numerical methods remain unchanged. The two competing explanations in the initial self-review remain open; measured event differences do not isolate traffic or establish a full author benchmark, LLM or video result. The V2 evidence remains separately bound as the actual instrumented eager measurement.

Revised producer SHA `15f52666e19e6795084ba21144812c63813ba31409627aec54a23a235c4fd04f`, tests SHA `b8628802746540cfe2a2cfa636056dfd777267f8dde3776d7bc3ac1855aafea5`.

Revised protocol SHA `737f25518d84724d4356e546662235b05655cc968785aa2e1f585ec77bf1e2d8`, fingerprint `6f533bbd6809c68a58a3853fab6fdf144c41c50a0bcd564ffe0246d2146af330`, private supervisor SHA `a9076e4b61f637a20d03ecd982962e80f9ad72efdbea4bce08093d14804394dd`. No ACTIVE registration or GPU execution occurred. Please independently recheck the revised exact artifacts.

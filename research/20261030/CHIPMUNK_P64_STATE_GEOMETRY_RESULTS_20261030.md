# Chipmunk global-P64 saved-state geometry

Two exposed development sources and one seed each. Dense and P64 share byte-identical initial noise; the four step-0 controls (flow and latent for each source) are byte-identical. Results below describe paired trajectories, not the isolated causal contribution of a step.

| Source | Denoising step | State | Error energy | Dense reference energy | Relative SSE | DC / error | Latent-time AC / error | Spatial AC / error |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 898 | 0 | flow | 0 | 3.20579e+06 | 0.000000 | — | — | — |
| 898 | 0 | latent | 0 | 1.77952e+06 | 0.000000 | — | — | — |
| 898 | 1 | flow | 143504 | 3.1959e+06 | 0.044903 | 0.016610 | 0.013833 | 0.969557 |
| 898 | 1 | latent | 1564.82 | 1.42109e+06 | 0.001101 | 0.016512 | 0.013772 | 0.969717 |
| 898 | 2 | flow | 103527 | 3.07153e+06 | 0.033705 | 0.002409 | 0.006787 | 0.990804 |
| 898 | 2 | latent | 8076.69 | 944961 | 0.008547 | 0.007672 | 0.010303 | 0.982025 |
| 898 | 3 | flow | 211328 | 3.04224e+06 | 0.069465 | 0.006033 | 0.005071 | 0.988896 |
| 898 | 3 | latent | 116111 | 1.14617e+06 | 0.101303 | 0.005747 | 0.006367 | 0.987886 |
| 96 | 0 | flow | 0 | 2.90521e+06 | 0.000000 | — | — | — |
| 96 | 0 | latent | 0 | 1.77552e+06 | 0.000000 | — | — | — |
| 96 | 1 | flow | 113414 | 2.95202e+06 | 0.038419 | 0.018293 | 0.010131 | 0.971576 |
| 96 | 1 | latent | 1238.2 | 1.40618e+06 | 0.000881 | 0.018183 | 0.010070 | 0.971748 |
| 96 | 2 | flow | 70145.6 | 2.73706e+06 | 0.025628 | 0.004054 | 0.006047 | 0.989900 |
| 96 | 2 | latent | 5391.87 | 896406 | 0.006015 | 0.011945 | 0.008307 | 0.979748 |
| 96 | 3 | flow | 124900 | 2.68901e+06 | 0.046448 | 0.006139 | 0.005949 | 0.987912 |
| 96 | 3 | latent | 71014.7 | 804386 | 0.088284 | 0.006397 | 0.007112 | 0.986491 |

For latent relative SSE, the source-equal means at denoising steps 0–3 are 0.000000, 0.000991, 0.007281, 0.094794. Dense reference energy changes across steps, so these relative values are not an equal-factor statement about absolute error growth. Zero error has undefined component fractions, shown as —.

[channel, latent_time, height, width]; 21 latent-time indices differ from the four denoising steps. T is temporal variation of each channel spatial mean; S includes spatial structure and its variation over latent time; neither is a direct subject motion or flicker metric.

The three component energies, all signed doubled cross terms, finite closure residuals and tolerances are retained per source/step/state in the [whitelist JSON](CHIPMUNK_P64_STATE_GEOMETRY_RESULTS_20261030.json). The observed small cross and closure terms are finite arithmetic results; they are not assumed exact.

Separately reviewed and registered matched high-only versus low-only reuse, with same source/seed/Dense controls and full free-generation states, can isolate expert-action effects; this is a proposal, not an executed scope.

The CPU scope actually exited zero, with 503666678 charged and completed input bytes and no outstanding read. No saved tensor was reread to prepare this report; source/sample limits remain two exposed image clusters, not held-out quality evidence.

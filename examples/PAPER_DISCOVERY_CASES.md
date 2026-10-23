# Three paper-to-discovery cases (worked research patterns)

These are **methodological reconstructions**, not claims that AutoResearcher reproduced any reported result. Read the original paper and code/config when using an operator in a registered experiment. The transferable object is the **discovery move**, not a prompt to copy the published algorithm.

## Case E: HyperMLP — discover freedom through an exact reparameterization

**Source:** Jiecheng Lu and Shihao Yang, *HyperMLP: An Integrated Perspective for Sequence Modeling*, ICML 2026. https://proceedings.mlr.press/v306/lu26aq.html ; https://arxiv.org/abs/2602.12601

**Old viewpoint:** attention as normalized similarity-based KV lookup.

**Exact scoped rewrite:** for one autoregressive attention head with q=xWq, K=XWk, V=XWv,

    o = softmax(x [Wq Wk^T X^T/sqrt(dk)]) [X Wv Wo]

The first bracket is a context-dependent input-to-history matrix; the second a history-to-output matrix. Both can include the current token if X includes x. Softmax couples hidden coordinates through normalization; this is not automatically a coordinatewise ReLU MLP.

**Structural opening:** hidden coordinates need not be permanently tied to individual positions. Explore dynamic mixing of feature and sequence axes and decoupling of routing and readout. HyperMLP/HyperGLU are new learned operators, **not** functions proven identical to original softmax attention.

**Minimal discriminator:** tiny deterministic single-head tensors verify the original/rewrite equality; compare position-tied vs learned mixing with matched budgets for any new prediction.

**PRD pattern:** exact algebra -> reveal hidden basis assumption -> change that assumption -> new architecture.

## Case T: MiTA — shared compression and routing from a mechanism taxonomy

**Source:** Qishuai Wen et al., *MiTA Attention: Efficient Fast-Weight Scaling via a Mixture of Top-k Activations*, arXiv:2602.01219 (official repository identifies NeurIPS 2026). https://arxiv.org/abs/2602.01219 ; https://github.com/QishuaiWen/MiTA

**Old viewpoint:** choose one efficient attention family such as compression or sparse retrieval.

**Taxonomy:** interpret growing KV context as a fast-weight memory; classify methods by compression versus routing, expert construction, selector reuse and cost. MiTA uses landmark queries to both produce global compressed memory and identify top-k original KV sets for deformable experts. This shares work that separate compression and routing schemes might duplicate.

**Hidden assumption changed:** global summary and selective detail must be computed by unrelated passes. One landmark-based probe can help build both.

**Minimal discriminator:** compare only-compression, only-routing, independent combination and shared construction under equal token budgets and end-to-end latency, including landmark-policy overhead. The exact attention-to-exponential-network rewrite may absorb 1/Z(q,K) into an apparent second-layer "weight"; this introduces query dependency and does not confer query-independent cached weights.

**PRD pattern:** common mathematical mechanism -> independent taxonomy axes -> identify reusable shared signal -> constructive hybrid with ablation.

## Case R: Frame2Frame — reduce a static task to a temporal solver

**Source:** Noam Rotstein et al., *Pathways on the Image Manifold: Image Editing via Video Generation*, CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Rotstein_Pathways_on_the_Image_Manifold_Image_Editing_via_Video_Generation_CVPR_2025_paper.html ; https://rotsteinnoam.github.io/Frame2Frame/

**Old viewpoint:** generate one edited image directly while jointly optimizing edit adherence and original-image preservation.

**Reformulation:** a vision-language model converts an image edit into a temporal-edit caption; an image-to-video model generates a sequence from the input image; a frame selector chooses a suitable edited state.

    edit(image, instruction)
      ~ select_frame(video_model(image, temporal_caption(instruction)))

**Structural opening:** a video generator contains continuity priors potentially useful for content-preserving edits. The path on an image manifold is an informative interpretation, not a verified exact geodesic or algebraic equivalence between editing and video generation.

**Minimal discriminator:** compare direct editing with trajectory generation and selection under matched generation + selection budgets. Check semantic target fidelity, source preservation, trajectory smoothness and failures when the edit cannot be represented as a plausible temporal transition.

**PRD pattern:** change the problem and use another mature solver -> specify readout and preserved constraints -> validate the actual end task.

## Cross-case synthesis

Three distinct moves:
1. **E:** identical focal computation, different representation; unlock basis or operator freedom.
2. **T:** shared abstraction of existing approaches; uncover missing combinations and reusable information.
3. **R:** non-equivalent problem reduction; exploit inductive bias of another solver.

Distinguish: *exact algebra* vs *conditional equality* vs *approximation* vs *task transfer* vs *analogy*. PRD seeds invite broad creativity; only experiment and novelty claims require corresponding evidence.

Suggested independent challenge: starting with a gated FFN F(x)=sum_i h_i(x)w_{d,i}, derive the output Gram geometry, then propose a new **pre-compute-available** signal and execution-compatible action. Compare against activation thresholding, output-aware oracle and dense complete generation. An output-aware oracle alone does not establish savings before hidden activation computation.

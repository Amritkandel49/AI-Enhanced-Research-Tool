
BERT’s pre‑training objective is a joint cross‑entropy over two tasks: masked language modeling (MLM) and next‑sentence prediction (NSP). For a token sequence (x=(x_1,\dots ,x_T)) and a set of masked positions (\mathcal{M}\subseteq{1,\dots ,T}), the MLM loss is

$$\mathcal{L}{	\text{MLM}}(x)= -\sum{t\in\mathcal{M}}\log P_	\theta(x_t\mid x_{\setminus t}),$$

where (P_	\theta) is the softmax output of the Transformer encoder parameterised by (	\theta). The NSP loss is

$$\mathcal{L}{	\text{NSP}}(x)= -\log P	\theta(	\text{next}\mid x),$$

and the total loss is (\mathcal{L}=\mathcal{L}{	\text{MLM}}+\mathcal{L}{	\text{NSP}}). The authors argue that the original BERT training schedule—limited steps, modest batch size, and a suboptimal learning‑rate schedule—leads to under‑training, i.e., the model does not fully explore the parameter space that would minimise (\mathcal{L}).

The paper’s mathematical contribution lies in a systematic re‑engineering of the optimisation pipeline. They employ the AdamW optimiser with update rule

$$heta_{t+1}= 	heta_t-lpha_trac{m_t}{\sqrt{v_t}+\epsilon}-lpha_t\lambda	heta_t,$$

where (lpha_t) follows a linear warm‑up followed by cosine decay, (m_t) and (v_t) are the first‑ and second‑moment estimates, and (\lambda) is a weight‑decay coefficient. Gradient clipping is applied by normalising the gradient vector to a maximum (L_2) norm (	au). Crucially, the authors replace NSP with a “dynamic masking” strategy: each training batch re‑samples the mask positions (\mathcal{M}) on the fly, ensuring that the model sees a richer distribution of masked tokens. They also increase the batch size (B) (e.g., (B=256) or larger) and extend the total number of training steps (S) (often to (Spprox 1,	\text{M})), thereby allowing the optimiser to converge to a lower (\mathcal{L}).

These modifications, while conceptually simple, yield a robust optimisation trajectory. By decoupling the learning‑rate schedule from the batch size, applying weight decay directly to the parameters, and removing the NSP objective (which has been shown to provide limited signal), the training process becomes more stable and efficient. Empirically, the resulting “Roberta” model surpasses all subsequent post‑BERT methods on standard benchmarks, demonstrating that careful tuning of the optimisation hyper‑parameters can unlock performance gains that were previously unattainable with the vanilla BERT recipe.

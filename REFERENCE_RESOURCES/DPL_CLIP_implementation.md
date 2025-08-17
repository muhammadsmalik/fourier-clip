algorithms.py:
class CLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model = clip.load(self.hparams['clip_backbone'])[0].float()

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print('Set self.clip_model.parameters.reguires_grad = False!')

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512  # 
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)
        
    def update(self, minibatches, unlabeled=None):
        return {'loss': 0}
    
    def predict(self, x):
        logits_per_image, _ = self.clip_model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)


⸻

hparams_registry.py:


🔹 For CLIP

elif algorithm == "CLIP":
    _hparam('prompt', 'class_name', lambda r: r.choice(['class_name', 'domain_name']))

	•	Defines a single hyperparameter:
	•	prompt: default = "class_name".
	•	Random choice = "class_name" or "domain_name".
	•	This controls how the text prompt is constructed. In algorithms/clip.py, you saw:

self.prompt = clip.tokenize(
    [f"a photo of a {c}" for c in hparams['class_names']]
).to(self.device)

That’s where this hparam comes in — it decides whether {c} is the class name or something else (like domain name).

⸻

🔹 For DPLCLIP

elif algorithm in ["DPLCLIP"]:
    _hparam('num_domain_tokens', 16, lambda r: int(r.choice([2, 4, 8, 16])))  
    _hparam('mlp_depth', 3, lambda r: int(r.choice([3])))
    _hparam('mlp_width', 512, lambda r: int(r.choice([256, 512])))
    _hparam('mlp_dropout', 0.1, lambda r: r.choice([0.0, 0.1]))

	•	Defines hyperparameters specific to the learnable prompt and auxiliary MLP in DPLCLIP.
	•	num_domain_tokens: how many prompt tokens are added.
	•	mlp_depth, mlp_width, mlp_dropout: architecture of the small MLP used in prompt learning.

⸻

🔹 Shared Training Params

These apply to both CLIP and DPLCLIP, depending on dataset:

if dataset in SMALL_IMAGES or algorithm in ["DPLCLIP"]:  
    _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
else:
    _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

if dataset in SMALL_IMAGES or algorithm in ["DPLCLIP"]:
    _hparam('weight_decay', 0., lambda r: 0.)
    _hparam('momentum', 0.1, lambda r: r.choice([0.0, 0.1, 0.2]))
else:
    _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))

	•	For CLIP (pure zero-shot), learning rate & weight decay technically don’t matter, since update() just returns {'loss': 0.}.
	•	For DPLCLIP, these do matter, since it trains the prompt learner.

⸻

✅ So summary for your question:
	•	For CLIP, the only relevant part here is the prompt hyperparameter.
	•	For DPLCLIP, all of num_domain_tokens, mlp_*, plus standard lr, momentum, and weight_decay apply.

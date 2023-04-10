import random
import textwrap

import evals
import evals.metrics



class Pattern(evals.Eval):
    def __init__(self,  test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.test_jsonl = test_jsonl


    def run(self, recorder):

        test_samples = evals.get_jsonl(self.test_jsonl)
        self.eval_all_samples(recorder, test_samples)

        # Record overall metrics
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }


    def eval_sample(self, test_sample, rng: random.Random):


        prompt = [
            {"role": "system", "content": "Solve the following math pattern problems. Please only give the answer"},
        ]

        for i, sample in enumerate([test_sample]):
            prompt += [{"role": "user", "content": sample["prompt"]}]


        evals.check_sampled_text(self.model_spec, prompt, expected=sample["completion"])
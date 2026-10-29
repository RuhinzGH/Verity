"""Run once while the Space image is built: load every model the app uses and compute the
SciFact corpus embeddings, so they are cached inside the image.

    python -m deploy.warm
"""
import time

from verity.pipeline import Verity

if __name__ == "__main__":
    t0 = time.perf_counter()
    corpus = Verity.scifact()
    corpus.check("Smoking is associated with an increased risk of lung cancer.")
    pasted = Verity([("mars.txt", "Mars is the fourth planet from the Sun. It has two small moons.")])
    pasted.check("Mars is the third planet from the Sun.")
    print(f"warm: verifier {corpus.verifier}, {time.perf_counter() - t0:.0f} s")

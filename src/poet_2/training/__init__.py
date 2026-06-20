"""Continued-training (continual pre-training) stack for PoET-2.

This subpackage adds the training machinery that the inference release does not ship:
the §8.2 masking/noise schedule (:mod:`poet_2.training.noise`), the §7 three-term loss
on the AA half of the 58-wide head (:mod:`poet_2.training.losses`), a homolog-set data
collator, and the train loop. Nothing in the existing model package is modified; the
released checkpoint is loaded and continue-trained.

Section references (e.g. §7, §8.2) point to ``poet2_train_spec.md``.
"""

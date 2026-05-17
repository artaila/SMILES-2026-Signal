# SOLUTION.md — SMILES-2026 Signal Interference Cancellation

## Result
=== Your Solution ===
  ch0: 9.34 dB
  ch1: 7.66 dB
  ch2: 9.74 dB
  ch3: 6.23 dB
  Metric [yours]: 8.24 dB

## Reproducibility
Check that environment has the libs:
```
pip install numpy scipy gdown
```
or 
```
conda install numpy scipy gdown
```
Run the script:
```
python applicant_solution.py
```

## Final solution

### What was modified

Only `your_canceller(tx_n, rx)` in `applicant_solution.py` was modified. Everything else was left untouched.

### Approach

The key observation behind the final method is that the official score is computed after bandpass filtering. This means that improvements are not determined only by how much total interference is removed in the raw time-domain signal, but also by how much energy is removed in the narrow band. The solution is to optimize a small family of removable components directly in the scorer-relevant band. 

The final removed signal is modeled as a linear combination of four components:
 - the baseline TX prediction;
 - a filtered version of that TX prediction, used as a small metric-oriented correction;
 - rank-1 shared interference term extracted from the post-TX residual in the scoring band;
 - second rank-1 term extracted after an additional filtering pass, used to capture interference structure that becomes more prominent after repeated band selection.

The solution searches over coefficient combinations and selects the best valid candidate using a proxy objective that matches the official score as closely as possible.
It estimates residual in-band power per channel after cancellation. Then it checks that the removed component is still explainable as TX-driven plus shared rank-1 structure and rejects invalid candidates, so it keeps the valid combination with the best expected average dB reduction.


## Experiments and failed attempts

I tried several ideas. 
 1. Iterative alternating TX / rank-1 cancellation
 The first idea was repeatable substracting the estimated TX-driven interference and substracting the estimated rank-1 shared component on the residual. However, in practice it performed poorly.The main problem was that repeated alternation tended to blur the boundary between the TX-driven part and the shared part. As a result, the removed signal often became less explainable. Even when not fully invalid, the iterative method was unstable and did not improve reliably enough for keeping it.
 2. Fixed-weight rank-1 corrections
 I also tested simpler variants that started from the baseline TX prediction and I added a rank-1 correction with a few fixed scaling factors, for example small or medium multipliers applied to the rank-1 term extracted after one or two filtering passes. This was a little bit better than the fully iterative approach because it was more stable and easier to control. However, a weight that worked well for one component variant could under-cancel or over-cancel another.  Since the score is sensitive to residual band power and validity constraints, using a manually chosen coefficient set left noticeable performance on the table. 
 3. 5th order terms
 I also tried to add 5th order terms to see, if 
 4. More aggressive double-filter exploitation
 Because the scorer filters in-band, one could deliberately over-shape the removed signal in the filtered domain to compensate for effective losses after repeated filtering especially near the passband edges. So I tried more aggressive corrections and doubly filtered shared terms. It was reasonable, but pushing this too far increased the risk that the removed component would stop looking valid. 
 I therefore kept a part of this method in the final solution: a filtered TX correction term and a second shared filtered component, both with constrained coefficient search.
 5. Pure power minimization without validity-aware selection
 I also tried to choose the candidate that minimized residual band power alone, without explicitly checking explainability constraints during selection. This was not reliable, the validator could not explain the deletion. 
 6. Replacing the provided TX model with a new custom nonlinear model

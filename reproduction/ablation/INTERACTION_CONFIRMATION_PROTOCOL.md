# MSEFPaper x ADown interaction confirmation

Frozen before the seed-1/2 replication runs. This experiment tests whether the
positive interaction discovered at seed 0 is reproducible; it is not a new
architecture search.

## Design

All four cells share the same YOLO11n-P2 parent graph and CIoU loss.

| MSEFPaper | full-path ADown | Cell |
|---|---|---|
| off | off | P2 |
| on | off | P2 + MSEFPaper |
| off | on | P2 + ADown |
| on | on | P2 + MSEFPaper + ADown |

- Seed 0 is the exploratory discovery seed and is reused without retraining.
- Seeds 1 and 2 are the predeclared replication seeds; all eight missing runs
  are trained from scratch.
- Dataset: fixed, format-aware deduplicated `DetectDataset_clean_v2`.
- Budget: true 60 epochs, 640 px, batch 16, AdamW, no AMP, no pretraining.
- Checkpoint selection: best validation mAP50:95 within the common 60-epoch
  budget. The test split remains untouched.
- Primary outcome: validation mAP50:95. Precision, recall and mAP50 are reported
  at the same selected epoch.

For each seed and metric, the interaction in percentage points is fixed as:

`100 * (cell_11 - cell_10 - cell_01 + cell_00)`

The interaction is called directionally replicated only if it is positive in
both held-out replication seeds. It is called practically stable only if the
mean held-out interaction is at least +0.30 percentage points. Because there
are only two held-out seeds, no null-hypothesis p-value will be reported.

If the interaction passes both rules, the joint method may advance to the
separate 200-epoch, three-seed confirmation against YOLO11n. If it fails, the
negative or inconclusive result is retained and the joint-effect claim is not
made.

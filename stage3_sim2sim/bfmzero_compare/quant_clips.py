"""Canonical set of GROUNDED near-ground clips for the quantitative analysis.

Each entry: (index, artifact_dir_name (with :v0), category, short label). Grounded
= lowest body within a few cm of the floor (verified by FK); floating-reference
retargets are excluded. Index = motion id in the BFM-Zero pkl.
"""
QUANT_CLIPS = [
    (0, "crouch_ff_start_180_R_003__A145_M:v0", "crouch", "crouch (stand->deep crouch)"),
    (1, "squat_001__A360:v0", "squat", "squat"),
    (2, "squat_001__A362_M:v0", "squat", "squat"),
    (3, "squat_002__A359:v0", "squat", "squat"),
    (4, "squat_002__A362:v0", "squat", "squat"),
    (5, "sit_on_chair_stop_R_001__A047:v0", "sit", "sit down on chair"),
    (6, "sit_on_chair_start_R_001__A167:v0", "sit", "stand up from chair"),
    (7, "sitting_legs_bend_arms_back_stop_001__A166_M:v0", "sit", "sit, legs bent, arms back"),
    (8, "sit_cross_legged_loop_R_001__A414:v0", "sit", "cross-legged floor sit"),
    (9, "idle_crawl_stop_003__A128:v0", "crawl", "crawl"),
]
ART = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed_full"

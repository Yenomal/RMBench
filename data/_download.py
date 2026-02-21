from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TianxingChen/RMBench",
    allow_patterns=["data/**"],
    local_dir=".",
    repo_type="dataset",
    resume_download=True,
)

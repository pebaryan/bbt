param(
  [string]$Data = "artifacts/datasets/tinystories/shards/data",
  [int]$Steps = 200000,
  [int]$BatchSize = 1,
  [int]$GradAccum = 2,
  [int]$SeqLen = 1024,
  [double]$Lr = 1e-4,
  [int]$DiffusionSteps = 64,
  [double]$MinMaskProb = 0.05,
  [double]$MaxMaskProb = 0.5,
  [switch]$UseSDPA,
  [switch]$UseDDP,
  [switch]$SmokeTest,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ExtraArgs
)

$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir = "artifacts/checkpoints/diffusion"
$out = Join-Path $outDir ("ckpt_diffusion-" + $ts + ".pt")
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$cmd = @(
  "python", "train_bitbyte_diffusion.py",
  "--data", $Data,
  "--steps", "$Steps",
  "--batch_size", "$BatchSize",
  "--grad_accum", "$GradAccum",
  "--seq_len", "$SeqLen",
  "--lr", "$Lr",
  "--diffusion_steps", "$DiffusionSteps",
  "--min_mask_prob", "$MinMaskProb",
  "--max_mask_prob", "$MaxMaskProb",
  "--out", $out
)

if ($UseSDPA) {
  $cmd += "--use_sdpa"
}
if ($UseDDP) {
  $cmd += "--ddp"
}
if ($SmokeTest) {
  $cmd += "--smoke_test"
}
if ($ExtraArgs) {
  $cmd += $ExtraArgs
}

Write-Host "Launching Diffusion run"
Write-Host "Output checkpoint: $out"
Write-Host ($cmd -join " ")
& $cmd[0] $cmd[1..($cmd.Length - 1)]

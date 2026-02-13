param(
  [string]$Data = "artifacts/datasets/tinystories/shards/data",
  [int]$Steps = 200000,
  [int]$BatchSize = 1,
  [int]$GradAccum = 2,
  [int]$SeqLen = 1024,
  [double]$Lr = 1e-4,
  [switch]$UseDDP,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ExtraArgs
)

$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir = "artifacts/checkpoints/mamba"
$out = Join-Path $outDir ("ckpt_mamba-" + $ts + ".pt")
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$cmd = @(
  "python", "train_mamba_new.py",
  "--data", $Data,
  "--steps", "$Steps",
  "--batch_size", "$BatchSize",
  "--grad_accum", "$GradAccum",
  "--seq_len", "$SeqLen",
  "--lr", "$Lr",
  "--out", $out
)

if ($UseDDP) {
  $cmd += "--ddp"
}
if ($ExtraArgs) {
  $cmd += $ExtraArgs
}

Write-Host "Launching Mamba run"
Write-Host "Output checkpoint: $out"
Write-Host ($cmd -join " ")
& $cmd[0] $cmd[1..($cmd.Length - 1)]

# リモート GPU サーバーで train_nnue.py を実行し、nnue.bin をコピーバックする
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$TrainArgs
)

$ErrorActionPreference = "Stop"

# === リモート設定 ===
$REMOTE_HOST = "fujita_ryohei@10.119.146.17"
$REMOTE_DIR  = "daiuchi_chappy_sho"
$REMOTE_PYTHON = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate && python"
$SSH_KEY     = "$env:USERPROFILE\.ssh\ed25519"
$SSH_OPTS    = @("-i", $SSH_KEY, "-o", "StrictHostKeyChecking=accept-new")

# === ローカルパス ===
$PROJECT_DIR = Split-Path -Parent $PSScriptRoot
$TOOLS_DIR   = $PSScriptRoot

# === 1. tools/*.py をリモートに転送 ===
Write-Host "=== Syncing tools/*.py to ${REMOTE_HOST} ===" -ForegroundColor Cyan
$pyFiles = Get-ChildItem "$TOOLS_DIR\*.py" | ForEach-Object { $_.FullName }
& scp @SSH_OPTS -q @pyFiles "${REMOTE_HOST}:~/${REMOTE_DIR}/tools/"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

# === 2. リモートで学習実行 ===
$argsStr = $TrainArgs -join " "
Write-Host "=== Running NNUE training on ${REMOTE_HOST} ===" -ForegroundColor Cyan
Write-Host "  args: $argsStr" -ForegroundColor Gray
& ssh @SSH_OPTS -t $REMOTE_HOST "cd ~/${REMOTE_DIR} && ${REMOTE_PYTHON} tools/train_nnue.py $argsStr"
if ($LASTEXITCODE -ne 0) { throw "Training failed (exit code $LASTEXITCODE)" }

# === 3. nnue.bin + nnue_model.pt をコピーバック ===
Write-Host "=== Copying nnue.bin back ===" -ForegroundColor Cyan
& scp @SSH_OPTS -q "${REMOTE_HOST}:~/${REMOTE_DIR}/nnue.bin" "$PROJECT_DIR\nnue.bin"
if ($LASTEXITCODE -ne 0) { throw "scp (nnue.bin) failed" }

& scp @SSH_OPTS -q "${REMOTE_HOST}:~/${REMOTE_DIR}/nnue_model.pt" "$PROJECT_DIR\nnue_model.pt"
if ($LASTEXITCODE -ne 0) { Write-Host "  Warning: nnue_model.pt copy failed (non-fatal)" -ForegroundColor Yellow }

$releaseDir = "$PROJECT_DIR\build\Release"
if (Test-Path $releaseDir) {
    Copy-Item "$PROJECT_DIR\nnue.bin" "$releaseDir\nnue.bin" -Force
    Write-Host "  -> build\Release\nnue.bin" -ForegroundColor Green
}

Write-Host "=== Done ===" -ForegroundColor Green

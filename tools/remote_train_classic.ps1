# リモート GPU サーバーで train_classic.py を実行し、linear.weights をコピーバックする
# Classic 学習はエンジン実行ファイルが必要なのでリモートでビルドも行う
param(
    [switch]$SkipBuild,
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
$SRC_DIR     = Join-Path $PROJECT_DIR "src"

# === 1. tools/*.py をリモートに転送 ===
Write-Host "=== [1/4] Syncing tools/*.py ===" -ForegroundColor Cyan
& ssh @SSH_OPTS $REMOTE_HOST "mkdir -p ~/${REMOTE_DIR}/tools"
$pyFiles = Get-ChildItem "$TOOLS_DIR\*.py" | ForEach-Object { $_.FullName }
& scp @SSH_OPTS -q @pyFiles "${REMOTE_HOST}:~/${REMOTE_DIR}/tools/"
if ($LASTEXITCODE -ne 0) { throw "scp (tools) failed" }

# === 2. ビルド ===
if (-not $SkipBuild) {
    Write-Host "=== [2/4] Syncing src/ and building kishi-to-classic ===" -ForegroundColor Cyan
    & ssh @SSH_OPTS $REMOTE_HOST "mkdir -p ~/${REMOTE_DIR}/src"
    $srcFiles = Get-ChildItem "$SRC_DIR\*" -Include "*.h","*.cpp" | ForEach-Object { $_.FullName }
    & scp @SSH_OPTS -q @srcFiles "${REMOTE_HOST}:~/${REMOTE_DIR}/src/"
    if ($LASTEXITCODE -ne 0) { throw "scp (src) failed" }
    & scp @SSH_OPTS -q "$PROJECT_DIR\CMakeLists.txt" "${REMOTE_HOST}:~/${REMOTE_DIR}/"
    if ($LASTEXITCODE -ne 0) { throw "scp (CMakeLists.txt) failed" }

    & ssh @SSH_OPTS $REMOTE_HOST "cd ~/${REMOTE_DIR} && cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 && cmake --build build --config Release --target kishi-to-classic -j`$(nproc) 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "Remote build failed" }
    Write-Host "  Build OK" -ForegroundColor Green
} else {
    Write-Host "=== [2/4] Skipping build (-SkipBuild) ===" -ForegroundColor Yellow
}

# === 3. 学習実行 ===
$argsStr = $TrainArgs -join " "
if (-not $argsStr) {
    $argsStr = "--kifu kifu/floodgate --engine build/kishi-to-classic --epochs 1"
}
Write-Host "=== [3/4] Running Classic training on ${REMOTE_HOST} ===" -ForegroundColor Cyan
Write-Host "  args: $argsStr" -ForegroundColor Gray
& ssh @SSH_OPTS -t $REMOTE_HOST "cd ~/${REMOTE_DIR} && ${REMOTE_PYTHON} tools/train_classic.py $argsStr"
if ($LASTEXITCODE -ne 0) { throw "Training failed (exit code $LASTEXITCODE)" }

# === 4. linear.weights をコピーバック ===
Write-Host "=== [4/4] Copying linear.weights back ===" -ForegroundColor Cyan
& scp @SSH_OPTS -q "${REMOTE_HOST}:~/${REMOTE_DIR}/linear.weights" "$PROJECT_DIR\linear.weights"
if ($LASTEXITCODE -ne 0) { throw "scp (linear.weights) failed" }

$releaseDir = "$PROJECT_DIR\build\Release"
if (Test-Path $releaseDir) {
    Copy-Item "$PROJECT_DIR\linear.weights" "$releaseDir\linear.weights" -Force
    Write-Host "  -> build\Release\linear.weights" -ForegroundColor Green
}

Write-Host "=== Done ===" -ForegroundColor Green

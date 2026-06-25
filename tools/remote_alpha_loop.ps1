# リモート GPU サーバーで alpha_train_loop.py (自己対局 → 学習 → 評価 → 昇格) を実行
# エンジンのビルド + 既存データの転送 + ループ実行 + best model のコピーバック
param(
    [switch]$SkipBuild,
    [switch]$SkipDataSync,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$LoopArgs
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
$WORK_DIR    = Join-Path $PROJECT_DIR "alpha_rl_work"

# === 1. tools/*.py をリモートに転送 ===
Write-Host "=== [1/6] Syncing tools/*.py ===" -ForegroundColor Cyan
& ssh @SSH_OPTS $REMOTE_HOST "mkdir -p ~/${REMOTE_DIR}/tools"
$pyFiles = Get-ChildItem "$TOOLS_DIR\*.py" | ForEach-Object { $_.FullName }
& scp @SSH_OPTS -q @pyFiles "${REMOTE_HOST}:~/${REMOTE_DIR}/tools/"
if ($LASTEXITCODE -ne 0) { throw "scp (tools) failed" }

# === 2. C++ ソース + CMakeLists.txt を転送してビルド ===
if (-not $SkipBuild) {
    Write-Host "=== [2/6] Syncing src/ and building kishi-to-alpha ===" -ForegroundColor Cyan
    & ssh @SSH_OPTS $REMOTE_HOST "mkdir -p ~/${REMOTE_DIR}/src"

    $srcFiles = Get-ChildItem "$SRC_DIR\*" -Include "*.h","*.cpp" | ForEach-Object { $_.FullName }
    & scp @SSH_OPTS -q @srcFiles "${REMOTE_HOST}:~/${REMOTE_DIR}/src/"
    if ($LASTEXITCODE -ne 0) { throw "scp (src) failed" }

    & scp @SSH_OPTS -q "$PROJECT_DIR\CMakeLists.txt" "${REMOTE_HOST}:~/${REMOTE_DIR}/"
    if ($LASTEXITCODE -ne 0) { throw "scp (CMakeLists.txt) failed" }

    Write-Host "  Building (cmake -DUSE_ONNXRUNTIME=ON) ..." -ForegroundColor Gray
    & ssh @SSH_OPTS $REMOTE_HOST @"
cd ~/${REMOTE_DIR}
# ONNX Runtime C++ SDK (GPU) — 初回のみダウンロード
ONNX_VER=1.18.1
ONNX_DIR=`$HOME/onnxruntime-linux-x64-gpu-`$ONNX_VER
if [ ! -d "`$ONNX_DIR" ]; then
  echo '  Downloading ONNX Runtime C++ SDK ...'
  wget -q https://github.com/microsoft/onnxruntime/releases/download/v`$ONNX_VER/onnxruntime-linux-x64-gpu-`$ONNX_VER.tgz -O /tmp/ort.tgz
  tar xzf /tmp/ort.tgz -C `$HOME && rm /tmp/ort.tgz
fi
export ONNXRUNTIME_ROOT=`$ONNX_DIR
export LD_LIBRARY_PATH=`$ONNX_DIR/lib:`$LD_LIBRARY_PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_ONNXRUNTIME=ON 2>&1
cmake --build build --config Release --target kishi-to-alpha -j`$(nproc) 2>&1
"@
    if ($LASTEXITCODE -ne 0) { throw "Remote build failed (exit code $LASTEXITCODE)" }
    Write-Host "  Build OK" -ForegroundColor Green
} else {
    Write-Host "=== [2/6] Skipping build (-SkipBuild) ===" -ForegroundColor Yellow
}

# === 3. 既存の自己対局データ + モデルを転送 ===
if (-not $SkipDataSync) {
    Write-Host "=== [3/6] Syncing existing alpha_rl_work/ ===" -ForegroundColor Cyan
    & ssh @SSH_OPTS $REMOTE_HOST "mkdir -p ~/${REMOTE_DIR}/alpha_rl_work/data ~/${REMOTE_DIR}/alpha_rl_work/models"

    # データファイル (.npz)
    $dataDir = Join-Path $WORK_DIR "data"
    if (Test-Path $dataDir) {
        $npzFiles = Get-ChildItem "$dataDir\*.npz" -ErrorAction SilentlyContinue
        if ($npzFiles) {
            Write-Host "  Uploading $($npzFiles.Count) data files ..." -ForegroundColor Gray
            $npzPaths = $npzFiles | ForEach-Object { $_.FullName }
            & scp @SSH_OPTS -q @npzPaths "${REMOTE_HOST}:~/${REMOTE_DIR}/alpha_rl_work/data/"
            if ($LASTEXITCODE -ne 0) { throw "scp (data) failed" }
        }
    }

    # モデルファイル
    $modelsDir = Join-Path $WORK_DIR "models"
    if (Test-Path $modelsDir) {
        $modelFiles = Get-ChildItem "$modelsDir\*" -Include "*.pt","*.onnx" -ErrorAction SilentlyContinue
        if ($modelFiles) {
            Write-Host "  Uploading $($modelFiles.Count) model files ..." -ForegroundColor Gray
            $modelPaths = $modelFiles | ForEach-Object { $_.FullName }
            & scp @SSH_OPTS -q @modelPaths "${REMOTE_HOST}:~/${REMOTE_DIR}/alpha_rl_work/models/"
            if ($LASTEXITCODE -ne 0) { throw "scp (models) failed" }
        }
    }

    # 学習ログ
    $logFile = Join-Path $WORK_DIR "train_log.jsonl"
    if (Test-Path $logFile) {
        & scp @SSH_OPTS -q "$logFile" "${REMOTE_HOST}:~/${REMOTE_DIR}/alpha_rl_work/"
    }

    Write-Host "  Sync OK" -ForegroundColor Green
} else {
    Write-Host "=== [3/6] Skipping data sync (-SkipDataSync) ===" -ForegroundColor Yellow
}

# === 4. RL ループ実行 ===
$argsStr = $LoopArgs -join " "
if (-not $argsStr) {
    $argsStr = "--engine build/kishi-to-alpha --resume --device auto --gpus 2"
}
Write-Host "=== [4/6] Running alpha_train_loop.py ===" -ForegroundColor Cyan
Write-Host "  args: $argsStr" -ForegroundColor Gray
& ssh @SSH_OPTS -t $REMOTE_HOST "cd ~/${REMOTE_DIR} && export LD_LIBRARY_PATH=`$HOME/onnxruntime-linux-x64-gpu-1.18.1/lib:`$LD_LIBRARY_PATH && ${REMOTE_PYTHON} tools/alpha_train_loop.py $argsStr"
if ($LASTEXITCODE -ne 0) { throw "RL loop failed (exit code $LASTEXITCODE)" }

# === 5. best model をコピーバック ===
Write-Host "=== [5/6] Copying best model back ===" -ForegroundColor Cyan
& scp @SSH_OPTS -q "${REMOTE_HOST}:~/${REMOTE_DIR}/alpha_rl_work/models/best.onnx" "$PROJECT_DIR\alpha_model.onnx"
if ($LASTEXITCODE -ne 0) { throw "scp (best.onnx) failed" }

& scp @SSH_OPTS -q "${REMOTE_HOST}:~/${REMOTE_DIR}/alpha_rl_work/models/best.pt" "$PROJECT_DIR\alpha_model.pt"
if ($LASTEXITCODE -ne 0) { Write-Host "  Warning: best.pt copy failed (non-fatal)" -ForegroundColor Yellow }

$releaseDir = "$PROJECT_DIR\build\Release"
if (Test-Path $releaseDir) {
    Copy-Item "$PROJECT_DIR\alpha_model.onnx" "$releaseDir\alpha_model.onnx" -Force
    Write-Host "  -> build\Release\alpha_model.onnx" -ForegroundColor Green
}

# === 6. 学習ログもコピーバック ===
Write-Host "=== [6/6] Copying train_log.jsonl back ===" -ForegroundColor Cyan
if (-not (Test-Path $WORK_DIR)) { New-Item -ItemType Directory -Path $WORK_DIR | Out-Null }
& scp @SSH_OPTS -q "${REMOTE_HOST}:~/${REMOTE_DIR}/alpha_rl_work/train_log.jsonl" "$WORK_DIR\train_log.jsonl"
if ($LASTEXITCODE -ne 0) { Write-Host "  Warning: train_log.jsonl copy failed (non-fatal)" -ForegroundColor Yellow }

Write-Host "=== Done ===" -ForegroundColor Green

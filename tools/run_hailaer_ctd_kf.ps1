$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\liuzi\AppData\Local\Temp\vibe-kanban\worktrees\4e29-ct-f-60-80s\CT_FGO"
$outRoot = "D:\Code\dataset\hailaer\inertail\ctd_kf_compare_20260301"
$manifest = Join-Path $outRoot "dataset_manifest.csv"
$summaryCsv = Join-Path $outRoot "summary_rmse.csv"
$runLog = Join-Path $outRoot "run.log"

$ctExe = "D:\Code\CT_FGO\bin\Release\ob_gins_ct.exe"
$kfExe = "D:\Code\KF-GINS\bin\Release\KF-GINS.exe"

if (-not (Test-Path $outRoot)) {
    New-Item -ItemType Directory -Path $outRoot | Out-Null
}

function Write-Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    $line | Tee-Object -FilePath $runLog -Append
}

Write-Log "Start MATLAB export..."
matlab -batch "addpath('$repoRoot/tools'); export_hailaer_mat_to_txt"

if (-not (Test-Path $manifest)) {
    throw "Manifest not found: $manifest"
}

"dataset_id,scheme,n,rmse_e_m,rmse_n_m,rmse_u_m,rmse_2d_m,rmse_3d_m" | Set-Content -Path $summaryCsv -Encoding UTF8

$rows = Import-Csv -Path $manifest
foreach ($r in $rows) {
    $datasetId = $r.dataset_id
    $datasetOut = $r.output_dir
    $imuFile = $r.imu_file
    $gnssFile = $r.gnss_file
    $rtkFile = $r.rtk_file
    $tStart = [double]$r.t_start
    $tEnd = [double]$r.t_end

    $init = Get-Content -Path $gnssFile -TotalCount 1
    $initParts = $init -split "\s+"
    $initLat = [double]$initParts[1]
    $initLon = [double]$initParts[2]
    $initH = [double]$initParts[3]

    $ctOut = Join-Path $datasetOut "ct_D_output"
    $kfOut = Join-Path $datasetOut "kf_output"
    New-Item -ItemType Directory -Path $ctOut -Force | Out-Null
    New-Item -ItemType Directory -Path $kfOut -Force | Out-Null

    $ctCfg = Join-Path $datasetOut "ob_gins_ct_D.yaml"
    @"
gnssfile: "$($gnssFile -replace '\\','/')"
outputpath: "$($ctOut -replace '\\','/')"
save_multi_imu: true

imu_main:
  type: "standard"
  file: "$($imuFile -replace '\\','/')"
  columns: 7
  rate_hz: 1000
  antlever: [0.0, 0.0, 0.0]
  imunoise:
    accel_noise: 25
    gyro_noise: 0.004
    accel_bias_rw: 5
    gyro_bias_rw: 1
    accel_corr_time: 3600.0
    gyro_corr_time: 3600.0

windows: 50
starttime: $("{0:F6}" -f $tStart)
endtime: $("{0:F6}" -f $tEnd)
aligntime: 3
kf_interval_sec: 0.1
num_iterations: 20
isearth: true

debug:
  enable: true
  level: 1

comparison:
  enable: false
"@ | Set-Content -Path $ctCfg -Encoding UTF8

    $kfCfg = Join-Path $datasetOut "kf-gins.yaml"
    @"
imupath: "$($imuFile -replace '\\','/')"
gnsspath: "$($gnssFile -replace '\\','/')"
outputpath: "$($kfOut -replace '\\','/')"
imudatalen: 7
imudatarate: 1000
starttime: $("{0:F6}" -f $tStart)
endtime: $("{0:F6}" -f $tEnd)
initpos: [ $("{0:F10}" -f $initLat), $("{0:F10}" -f $initLon), $("{0:F4}" -f $initH) ]
initvel: [ 0.0, 0.0, 0.0 ]
initatt: [ 0.0, 0.0, 0.0 ]
initgyrbias: [ 0, 0, 0 ]
initaccbias: [ 0, 0, 0 ]
initgyrscale: [ 0, 0, 0 ]
initaccscale: [ 0, 0, 0 ]
initposstd: [ 2.0, 2.0, 5.0 ]
initvelstd: [ 0.5, 0.5, 0.5 ]
initattstd: [ 5.0, 5.0, 30.0 ]
imunoise:
  arw: [0.24, 0.24, 0.24]
  vrw: [0.24, 0.24, 0.24]
  gbstd: [50.0, 50.0, 50.0]
  abstd: [250.0, 250.0, 250.0]
  gsstd: [1000.0, 1000.0, 1000.0]
  asstd: [1000.0, 1000.0, 1000.0]
  corrtime: 1.0
antlever: [ 0.0, 0.0, 0.0 ]
"@ | Set-Content -Path $kfCfg -Encoding UTF8

    Write-Log "[$datasetId] Running CT_D..."
    cmd /c "`"$ctExe`" `"$ctCfg`" > `"$((Join-Path $datasetOut "ct_run.log"))`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
        throw "CT_D failed for $datasetId with code $LASTEXITCODE"
    }

    Write-Log "[$datasetId] Running KF..."
    cmd /c "`"$kfExe`" `"$kfCfg`" > `"$((Join-Path $datasetOut "kf_run.log"))`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
        throw "KF failed for $datasetId with code $LASTEXITCODE"
    }

    $ctTraj = Join-Path $ctOut "ct_trajectory.txt"
    $kfNav = Join-Path $kfOut "KF_GINS_Navresult.nav"
    if ((Test-Path $ctTraj) -and (Test-Path $kfNav) -and (Test-Path $rtkFile)) {
        $evalLines = python (Join-Path $repoRoot "tools\eval_nav_rmse.py") --ct $ctTraj --kf $kfNav --truth $rtkFile
        $evalFile = Join-Path $datasetOut "rmse_eval.txt"
        $evalLines | Set-Content -Path $evalFile -Encoding UTF8
        foreach ($ln in $evalLines) {
            if ($ln -match "^(CT_D|KF),") {
                "$datasetId,$ln" | Add-Content -Path $summaryCsv -Encoding UTF8
            }
        }
    } else {
        Write-Log "[$datasetId] Missing output files for evaluation."
    }
}

Write-Log "Done. Summary: $summaryCsv"

# Prueba varias semillas con el dataset oficial y 100 generaciones, para elegir la semilla
# por defecto de B y de C. Cada profundidad corre en un proceso aparte, igual que el
# benchmark oficial. B y C usan generadores distintos (mt19937_64 y Xoshiro), asi que la
# misma semilla da corridas distintas: se elige una semilla para cada implementacion.
#
# Uso, desde PowerShell en C:\proyecto, con B compilado en x64-Release:
#   powershell -ExecutionPolicy Bypass -File .\probar_semillas.ps1          (B y C)
#   ... -SoloC                 solo C (mucho mas rapido que B)
#   ... -SoloB                 solo B
#   ... -Seeds 1,2,3           otras semillas
#   ... -Depths 12             solo algunas profundidades
#   ... -Carpeta semillas_20261126_1700
#                              continuar una prueba interrumpida: salta lo ya medido
#
# Orden: primero todas las semillas en profundidad 7, luego 9, luego 12, asi que los
# resultados de 7 y 9 salen pronto. Cada corrida se guarda al terminar; si se corta,
# se continua con -Carpeta. Al final imprime un resumen ordenado por tiempo total.
#
# Tiempo aproximado con 100 generaciones (estimado a partir de 10 generaciones):
# C, unos 5-10 minutos por semilla; B, de media hora a una hora por semilla (sobre todo
# profundidad 12). No uses la computadora mientras mide y conecta el cargador.

param(
    [string]$B = "GP_CPP20_B\out\build\x64-Release\gp_b.exe",
    [string]$Julia = "julia",
    [string]$Data = "datos\dataset3a.csv",
    [int[]]$Seeds = @(1, 2, 3, 4, 5, 7, 11, 42, 123, 2026),
    [int[]]$Depths = @(7, 9, 12),
    [int]$Generations = 100,
    [string]$Carpeta = "",
    [switch]$SoloB,
    [switch]$SoloC
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$jl = @("--startup-file=no", "-O3")
$needed = @($Data)
if (-not $SoloC) { $needed += $B }
if (-not $SoloB) { $needed += "GP_JL_C\main.jl" }
foreach ($p in $needed) {
    if (-not (Test-Path $p)) { throw "No se encontro: $p" }
}
$Data = (Resolve-Path $Data).Path

$impls = @()
if (-not $SoloC) {
    $impls += @{ Impl = "B"; Cmd = (Resolve-Path $B).Path; Pre = @() }
}
if (-not $SoloB) {
    $impls += @{ Impl = "C"; Cmd = $Julia; Pre = $jl + (Resolve-Path "GP_JL_C\main.jl").Path }
}

if ($Carpeta -eq "") { $Carpeta = "semillas_" + (Get-Date -Format "yyyyMMdd_HHmm") }
if (-not (Test-Path $Carpeta)) { New-Item -ItemType Directory -Path $Carpeta | Out-Null }
$file = Join-Path $Carpeta "semillas.csv"
if (-not (Test-Path $file)) {
    "impl,seed,depth,generations,elapsed_seconds,best_mse" | Out-File -FilePath $file -Encoding utf8
}
$done = @{}
foreach ($r in @(Import-Csv $file)) {
    $done["$($r.impl),$($r.seed),$($r.depth),$($r.generations)"] = $true
}
Write-Host "Resultados en: $Carpeta"
Write-Host ("Semillas: {0}; profundidades: {1}; {2} generaciones" -f ($Seeds -join ', '),
            ($Depths -join ', '), $Generations)

$start = Get-Date
foreach ($d in $Depths) {
    foreach ($s in $Seeds) {
        foreach ($v in $impls) {
            if ($done.ContainsKey("$($v.Impl),$s,$d,$Generations")) { continue }
            $argList = $v.Pre + @("--bench-gp", "--data=$Data", "--population=5000",
                                  "--depths=$d", "--generations=$Generations", "--seed=$s")
            $lines = @(& $v.Cmd @argList)
            if ($LASTEXITCODE -ne 0) { throw "Fallo $($v.Impl) con la semilla $s en profundidad $d" }
            $f = $lines[1] -split ','
            $row = "$($v.Impl),$s,$d,$Generations,$($f[3]),$($f[4])"
            $row | Out-File -FilePath $file -Append -Encoding utf8
            Write-Host ("  {0}   (llevamos {1:N0} min)" -f $row, ((Get-Date) - $start).TotalMinutes)
        }
    }
}

# Resumen: una fila por implementacion y semilla, ordenado por tiempo total.
$rows = @(Import-Csv $file | Where-Object { [int]$_.generations -eq $Generations })
$summary = foreach ($g in ($rows | Group-Object impl, seed)) {
    $byDepth = @{}
    foreach ($r in $g.Group) { $byDepth[[int]$r.depth] = $r }
    $o = [ordered]@{ impl = $g.Group[0].impl; seed = [int]$g.Group[0].seed }
    $total = 0.0
    $complete = $true
    foreach ($d in $Depths) {
        if ($byDepth.ContainsKey($d)) {
            $t = [double]$byDepth[$d].elapsed_seconds
            $total += $t
            $o["s_$d"] = [math]::Round($t, 1)
            $o["mse_$d"] = $byDepth[$d].best_mse
        } else {
            $complete = $false
            $o["s_$d"] = ""
            $o["mse_$d"] = ""
        }
    }
    $o["total_s"] = if ($complete) { [math]::Round($total, 1) } else { "" }
    [pscustomobject]$o
}
$sortKey = @{ Expression = { if ($_.total_s -eq "") { [double]::MaxValue } else { [double]$_.total_s } } }
$summary = @($summary | Sort-Object impl, $sortKey)
$summary | Format-Table -AutoSize
$summary | Export-Csv (Join-Path $Carpeta "resumen.csv") -NoTypeInformation -Encoding UTF8

foreach ($impl in @("B", "C")) {
    $best = @($summary | Where-Object { $_.impl -eq $impl -and $_.total_s -ne "" }) | Select-Object -First 1
    if ($best) {
        Write-Host ("Semilla mas rapida de {0}: {1} ({2} s en total)" -f $impl, $best.seed, $best.total_s)
    }
}
$minutes = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host "`nListo en $minutes minutos. Pasame $Carpeta\resumen.csv."

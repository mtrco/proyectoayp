# Verifica el ultimo cambio en C (lista postfija y cache al reusarse) y mide cuanto ahorra.
# Corre las pruebas y despues el GP completo de C con y sin el cambio
# (--no-tape --reuse-min-size=0 da la version anterior). El mejor MSE de CADA generacion
# debe ser IGUAL en las dos; solo debe cambiar el tiempo.
#
# Uso, desde PowerShell en C:\proyecto:
#   .\verificar_cambios.ps1                 pruebas + GP en profundidades 7, 9 y 12 (~10 min)
#   .\verificar_cambios.ps1 -SinMedir       solo pruebas
#   .\verificar_cambios.ps1 -Depths 7       solo la profundidad 7
#
# No uses la computadora mientras mide. Resultados en verificacion_C_AAAAMMDD_HHMM.

param(
    [string]$Julia = "julia",
    [switch]$SinMedir,
    [int[]]$Depths = @(7, 9, 12),
    [int]$Generations = 100,
    [int]$Seed = 5
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$flags = @("--startup-file=no", "-O3")

Write-Host "`n[1] Pruebas de C"
& $Julia @flags "GP_JL_C\main.jl" --test
if ($LASTEXITCODE -ne 0) { throw "Fallan las pruebas de C" }

Write-Host "`n[2] Verificacion cruzada A <-> C"
& $Julia @flags "GP_JL_C\tools\cross_check.jl"
if ($LASTEXITCODE -ne 0) { throw "cross_check encontro diferencias entre A y C" }

Write-Host "`n[3] Verificacion A, C <-> B"
$gpb = "GP_CPP20_B\out\build\x64-release\gp_b.exe"
if (Test-Path $gpb) {
    & $Julia @flags "GP_JL_C\tools\verificar_con_b.jl" (Resolve-Path $gpb).Path
    if ($LASTEXITCODE -ne 0) { throw "verificar_con_b encontro diferencias" }
} else {
    Write-Host "  (no se encontro $gpb; compila B en x64-Release para esta prueba)"
}

if ($SinMedir) { Write-Host "`nPruebas correctas."; exit 0 }

Write-Host "`n[4] GP completo de C: sin el cambio (antes) contra con el cambio (ahora)"
Write-Host "    Poblacion 5000, $Generations generaciones, semilla $Seed, dataset3a, un proceso por corrida."
$data = (Resolve-Path "datos\dataset3a.csv").Path
$dir = Join-Path $PSScriptRoot ("verificacion_C_" + (Get-Date -Format "yyyyMMdd_HHmm"))
New-Item -ItemType Directory -Path $dir | Out-Null
$versions = @(
    @{ Name = "antes"; Extra = @("--no-tape", "--reuse-min-size=0") },
    @{ Name = "ahora"; Extra = @() }
)
$summary = @()
foreach ($d in $Depths) {
    $res = @{}
    foreach ($v in $versions) {
        $hist = Join-Path $dir "historial_$($v.Name)_$d.csv"
        Write-Host "== C $($v.Name), profundidad $d $($v.Extra -join ' ')"
        $argList = $flags + @("GP_JL_C\main.jl", "--bench-gp", "--data=$data", "--population=5000",
                              "--depths=$d", "--generations=$Generations", "--seed=$Seed",
                              "--history=$hist") + $v.Extra
        # El programa muestra en vivo (por stderr) el mejor MSE de cada generacion.
        $output = @(& $Julia @argList)
        if ($LASTEXITCODE -ne 0) { throw "Fallo el GP de C ($($v.Name)) en profundidad $d" }
        $fields = $output[1] -split ','
        $res[$v.Name] = @{ Seconds = [double]$fields[3]; Mse = $fields[4]; History = @(Get-Content $hist) }
        Write-Host ("  {0:N1} s   mejor MSE {1}" -f $res[$v.Name].Seconds, $res[$v.Name].Mse)
    }
    $sameHistory = $null -eq (Compare-Object $res["antes"].History $res["ahora"].History)
    $summary += [pscustomobject]@{
        profundidad       = $d
        antes_s           = [math]::Round($res["antes"].Seconds, 1)
        ahora_s           = [math]::Round($res["ahora"].Seconds, 1)
        veces_mas_rapido  = [math]::Round($res["antes"].Seconds / $res["ahora"].Seconds, 2)
        mse_igual         = $res["antes"].Mse -eq $res["ahora"].Mse
        historial_igual   = $sameHistory
        mejor_mse         = $res["ahora"].Mse
    }
}
$summary | Export-Csv -Path (Join-Path $dir "resumen.csv") -NoTypeInformation -Encoding utf8
Write-Host ""
$summary | Format-Table -AutoSize | Out-String | Write-Host
if (@($summary | Where-Object { -not $_.historial_igual }).Count -gt 0) {
    Write-Host "ATENCION: el MSE de alguna generacion cambio. El cambio NO debe alterar el resultado." -ForegroundColor Red
    exit 1
}
Write-Host "El MSE de todas las generaciones es igual con y sin el cambio. Archivos en $dir"

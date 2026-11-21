# Pruebas con el dataset oficial de la competencia (datos\dataset3a.csv) en ESTE equipo:
#   1. correctitud: pruebas de B y C, y verificacion cruzada A <-> C y A, C <-> B con dataset3a;
#   2. velocidad del GP completo: variantes de B y de C (todas deben dar el MISMO MSE).
#
# Uso, desde PowerShell en C:\proyecto, con B compilado en x64-Release:
#   powershell -ExecutionPolicy Bypass -File .\probar_dataset3a.ps1
#   ... -Rapido          solo profundidades 7 y 9 y una semilla (para probar que todo corre)
#   ... -Seeds 1,2,3     semillas (por defecto 1 y 2)
#   ... -SoloB / -SoloC  medir solo una implementacion
#   ... -SinVerificar    saltar las pruebas de correctitud
#
# Con 961 filas el GP tarda bastante mas que con ejemplo.csv: en profundidad 12 cada corrida
# de B tarda unos 3-4 minutos. Sin -Rapido, todo puede tardar 30-60 minutos. No uses la
# computadora mientras mide (cierra el navegador y conecta el cargador).

param(
    [string]$B = "GP_CPP20_B\out\build\x64-Release\gp_b.exe",
    [string]$Julia = "julia",
    [string]$Data = "datos\dataset3a.csv",
    [int[]]$Seeds = @(1, 2),
    [int]$Generations = 10,
    [switch]$Rapido,
    [switch]$SoloB,
    [switch]$SoloC,
    [switch]$SinVerificar
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$jl = @("--startup-file=no", "-O3")
foreach ($p in @($B, $Data, "GP_JL_A\main.jl", "GP_JL_C\main.jl")) {
    if (-not (Test-Path $p)) { throw "No se encontro: $p" }
}
$B = (Resolve-Path $B).Path
$Data = (Resolve-Path $Data).Path
$rootA = (Resolve-Path "GP_JL_A").Path
$depths = "7,9,12"
if ($Rapido) {
    $depths = "7,9"
    $Seeds = @($Seeds[0])
}
$start = Get-Date
$out = "pruebas_dataset3a_" + (Get-Date -Format "yyyyMMdd_HHmm")
New-Item -ItemType Directory -Path $out | Out-Null
Write-Host "Resultados en: $out"

# Huella (SHA-256) de los datasets oficiales, para el reporte.
Get-ChildItem "datos\*.csv" | Get-FileHash -Algorithm SHA256 |
    Select-Object @{ n = "archivo"; e = { Split-Path $_.Path -Leaf } }, Hash |
    Export-Csv "$out\sha256_datasets.csv" -NoTypeInformation -Encoding UTF8

if (-not $SinVerificar) {
    Write-Host "`n[1] Pruebas internas de B y C"
    & $B --test
    if ($LASTEXITCODE -ne 0) { throw "Fallan las pruebas de B" }
    & $Julia @jl "GP_JL_C\main.jl" --test
    if ($LASTEXITCODE -ne 0) { throw "Fallan las pruebas de C" }

    Write-Host "`n[2] Verificacion cruzada A <-> C con dataset3a"
    & $Julia @jl "GP_JL_C\tools\cross_check.jl" $rootA $Data
    if ($LASTEXITCODE -ne 0) { throw "cross_check encontro diferencias entre A y C" }

    Write-Host "`n[3] Verificacion A, C <-> B con dataset3a"
    & $Julia @jl "GP_JL_C\tools\verificar_con_b.jl" $B $rootA $Data
    if ($LASTEXITCODE -ne 0) { throw "verificar_con_b encontro diferencias" }
}

$mainC = @("GP_JL_C\main.jl")
$variants = @()
if (-not $SoloC) {
    $variants += @{ Impl = "B"; Name = "B";             Cmd = $B;     Pre = @();         Extra = @() }
    $variants += @{ Impl = "B"; Name = "B_sin_orden";   Cmd = $B;     Pre = @();         Extra = @("--no-order") }
}
if (-not $SoloB) {
    $variants += @{ Impl = "C"; Name = "C";             Cmd = $Julia; Pre = $jl + $mainC; Extra = @() }
    $variants += @{ Impl = "C"; Name = "C_sin_orden";   Cmd = $Julia; Pre = $jl + $mainC; Extra = @("--no-order") }
    $variants += @{ Impl = "C"; Name = "C_cache2048";   Cmd = $Julia; Pre = $jl + $mainC; Extra = @("--cache-mb=2048") }
    $variants += @{ Impl = "C"; Name = "C_cache4096";   Cmd = $Julia; Pre = $jl + $mainC; Extra = @("--cache-mb=4096") }
}

$file = "$out\gp_variantes.csv"
"impl,variante,seed,depth,elapsed_seconds,best_mse" | Out-File -FilePath $file -Encoding utf8
Write-Host "`n[4] GP completo con dataset3a: poblacion 5000, $Generations generaciones, profundidades $depths"
foreach ($s in $Seeds) {
    foreach ($v in $variants) {
        $argList = $v.Pre + @("--bench-gp", "--data=$Data", "--population=5000", "--depths=$depths",
                              "--generations=$Generations", "--seed=$s") + $v.Extra
        $lines = @(& $v.Cmd @argList)
        if ($LASTEXITCODE -ne 0) { throw "Fallo $($v.Name) con la semilla $s" }
        foreach ($line in ($lines | Select-Object -Skip 1)) {
            $f = $line -split ','
            $row = "$($v.Impl),$($v.Name),$s,$($f[0]),$($f[3]),$($f[4])"
            $row | Out-File -FilePath $file -Append -Encoding utf8
            Write-Host "  $row"
        }
    }
}

Write-Host "`n[5] Resumen: mediana de las semillas (segundos)"
$rows = Import-Csv $file
$summary = $rows | Group-Object variante, depth | ForEach-Object {
    $t = @($_.Group | ForEach-Object { [double]$_.elapsed_seconds } | Sort-Object)
    $mid = [int][math]::Floor(($t.Count - 1) / 2)
    if ($t.Count % 2 -eq 1) { $median = $t[$mid] } else { $median = ($t[$mid] + $t[$mid + 1]) / 2 }
    [pscustomobject]@{
        variante  = $_.Group[0].variante
        depth     = [int]$_.Group[0].depth
        mediana_s = [math]::Round($median, 2)
        best_mse  = (($_.Group | ForEach-Object { $_.best_mse }) -join " / ")
    }
}
$summary | Sort-Object depth, variante | Format-Table -AutoSize
$summary | Sort-Object depth, variante | Export-Csv "$out\resumen.csv" -NoTypeInformation -Encoding UTF8

# Todas las variantes de una misma implementacion deben dar el mismo MSE.
$distinct = $rows | Group-Object impl, seed, depth | Where-Object {
    @($_.Group | Select-Object -ExpandProperty best_mse -Unique).Count -gt 1
}
if ($distinct) {
    Write-Warning "Hay variantes con MSE distinto (deberia ser identico): revisa $file"
} else {
    Write-Host "OK: dentro de B y dentro de C, todas las variantes dieron el mismo MSE."
}
$minutes = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host "`nListo en $minutes minutos. Pasame la carpeta $out."
